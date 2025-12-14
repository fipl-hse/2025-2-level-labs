"""
Lab 4
"""

# pylint: disable=unused-argument, super-init-not-called, unused-private-member, duplicate-code, unused-import
import json

from lab_1_keywords_tfidf.main import check_positive_int
from lab_3_generate_by_ngrams.main import BackOffGenerator, NGramLanguageModel, TextProcessor

NGramType = tuple[int, ...]
"Type alias for NGram."

class TriePrefixNotFoundError(Exception):
    """
    Raised when the prefix required for transition is missing in the trie
    """


class EncodingError(Exception):
    """
    Raised when text encoding fails due to invalid input or processing error
    """


class DecodingError(Exception):
    """
    Raised when text decoding fails due to invalid input or processing error
    """


class IncorrectNgramError(Exception):
    """
    Raised when the N-gram size does not meet the requirements
    """


class MergeTreesError(Exception):
    """
    Raised when merging trees is impossible
    """

class WordProcessor(TextProcessor):
    """
    Handle text tokenization, encoding and decoding at word level.

    Inherits from TextProcessor but reworks logic to work with words instead of letters.
    """

    #: Special token to separate sentences
    _end_of_sentence_token: str

    def __init__(self, end_of_sentence_token: str) -> None:
        """
        Initialize an instance of SentenceStorage.

        Args:
            end_of_sentence_token (str): A token denoting sentence boundary
        """
        super().__init__(end_of_sentence_token)
        self._end_of_sentence_token = end_of_sentence_token

    def encode_sentences(self, text: str) -> tuple:
        """
        Encode text and split into sentences.

        Encodes text and returns a tuple of sentence sequences, where each sentence
        is represented as a tuple of word IDs. Sentences are separated by the
        end_of_sentence_token in the encoded text.

        Args:
            text (str): Original text to encode

        Returns:
            tuple: Tuple of encoded sentences, each as a tuple of word IDs
        """
        if not isinstance(text, str):
            raise EncodingError('Invalid input: text must be a string')
        sentences = []
        sentence = []

        for token in super().encode(text):
            sentence.append(token)
            if token == 0:
                if sentence:
                    sentences.append(tuple(sentence))
                    sentence.clear()

        if sentence:
            sentences.append(tuple(sentence))

        return tuple(sentences)

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): An element to put into storage

        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """
        if not isinstance(element, str) or not element:
            return
        if element not in self._storage:
            self._storage[element] = len(self._storage)


    def _postprocess_decoded_text(self, decoded_corpus: tuple[str, ...]) -> str:
        """
        Convert decoded sentence into the string sequence.

        Special symbols (end_of_sentence_token) separate sentences.
        The first letter is capitalized, resulting sequence must end with a full stop.

        Args:
            decoded_corpus (tuple[str, ...]): A tuple of decoded words

        Returns:
            str: Resulting text
        """
        if not isinstance(decoded_corpus, tuple) or not decoded_corpus:
            raise DecodingError("Invalid input: decoded_corpus must be a non-empty tuple")

        processed_text = []
        sentence = []

        for token in decoded_corpus:
            if token == self._end_of_sentence_token:
                if sentence:
                    processed_text.append(" ".join(sentence).capitalize())
                    sentence.clear()
            else:
                sentence.append(token)

        if sentence:
            processed_text.append(" ".join(sentence).capitalize())

        if not processed_text:
            raise DecodingError("Postprocessing resulted in empty output")

        return '. '.join(processed_text) + "."

    def _tokenize(self, text: str) -> tuple[str, ...]:
        """
        Tokenize text into words, separating sentences with special token.

        Punctuation and digits are removed from words.
        Sentences are separated by the end_of_sentence_token.

        Args:
            text (str): Original text

        Returns:
            tuple[str, ...]: Tokenized text as words
        """
        if not isinstance(text, str) or not text:
            raise EncodingError("Invalid input: text must be a non-empty string")

        tokens = []

        for word in text.lower().split():
            punctuation_end = False

            if any(word.endswith(char) for char in "?!."):
                punctuation_end = True

            cleaned_word = "".join(symbol for symbol in word if symbol.isalpha())

            if cleaned_word:
                tokens.append(cleaned_word)

                if punctuation_end:
                    tokens.append(self._end_of_sentence_token)

        if not tokens:
            raise EncodingError("Tokenization resulted in empty output")

        return tuple(tokens)



class TrieNode:
    """
    Node type for PrefixTrie.
    """

    #: Saved item in current TrieNode
    __name: int | None
    #: Additional payload to store in TrieNode
    _value: float
    #: Children nodes
    _children: list["TrieNode"]

    def __init__(self, name: int | None = None, value: float = 0.0) -> None:
        """
        Initialize a Trie node.

        Args:
            name (int | None, optional): The name of the node.
            value (float, optional): The value stored in the node.
        """
        self.__name = name
        self._value = value
        self._children = []

    def __bool__(self) -> bool:
        """
        Define the boolean value of the node.

        Returns:
            bool: True if node has at least one child, False otherwise.
        """
        return bool(self._children)

    def __str__(self) -> str:
        """
        Return a string representation of the N-gram node.

        Returns:
            str: String representation showing node data and frequency.
        """
        return f"TrieNode(name={self.get_name()}, value={self.get_value()})"

    def add_child(self, item: int) -> None:
        """
        Add a new child node with the given item.

        Args:
            item (int): Data value for the new child node.
        """
        self._children.append(TrieNode(item))

    def get_children(self, item: int | None = None) -> tuple["TrieNode", ...]:
        """
        Get the tuple of child nodes or one child.

        Args:
            item (int | None, optional): Special data to find special child

        Returns:
            tuple["TrieNode", ...]: Tuple of child nodes.
        """
        if item is None:
            return tuple(self._children)

        return tuple(
            child for child in self._children if child.get_name() == item
        )

    def get_name(self) -> int | None:
        """
        Get the data stored in the node.

        Returns:
            int | None: TrieNode data.
        """
        return self.__name

    def get_value(self) -> float:
        """
        Get the value of the node.

        Returns:
            float: Frequency value.
        """
        return self._value

    def set_value(self, new_value: float) -> None:
        """
        Set the value of the node

        Args:
            new_value (float): New value to store.
        """
        self._value = new_value

    def has_children(self) -> bool:
        """
        Check whether the node has any children.

        Returns:
            bool: True if node has at least one child, False otherwise.
        """
        return bool(self)

class PrefixTrie:
    """
    Prefix tree for storing token sequences.
    """

    #: Initial state of the tree
    _root: TrieNode

    def __init__(self) -> None:
        """
        Initialize an empty PrefixTrie.
        """
        self._root = TrieNode()

    def clean(self) -> None:
        """
        Clean the whole tree.
        """
        self._root = TrieNode()

    def fill(self, encoded_corpus: tuple[NGramType]) -> None:
        """
        Fill the trie based on an encoded_corpus of tokens.

        Args:
            encoded_corpus (tuple[NGramType]): Tokenized corpus.
        """
        if not isinstance(encoded_corpus, tuple):
            raise ValueError

        self.clean()

        for sequence in encoded_corpus:
            self._insert(sequence)

    def get_prefix(self, prefix: NGramType) -> TrieNode:
        """
        Find the node corresponding to a prefix.

        Args:
            prefix (NGramType): Prefix to find trie by.

        Returns:
            TrieNode: Found TrieNode by prefix
        """
        current_node = self._root

        for item in prefix:
            matching_children = current_node.get_children(item)
            if not matching_children:
                raise TriePrefixNotFoundError(f"Prefix {prefix} not found in the trie")
            current_node = matching_children[0]

        return current_node
    def suggest(self, prefix: NGramType) -> tuple:
        """
        Return all sequences in the trie that start with the given prefix.

        Args:
            prefix (NGramType): Prefix to search for.

        Returns:
            tuple: Tuple of all token sequences that begin with the given prefix.
                                   Empty tuple if prefix not found.
        """

        try:
            start_node = self.get_prefix(prefix)
        except TriePrefixNotFoundError:
            return tuple()

        start_node = self.get_prefix(prefix)
        results = []
        queue = [(tuple(prefix), start_node)]
        prefix_len = len(prefix)

        while queue:
            current_prefix, node = queue.pop(0)

            children = node.get_children()
            if not children:

                if not len(current_prefix) > prefix_len:
                    continue

                results.append(current_prefix)

            for child in children:
                name = child.get_name()
                if name is not None:
                    queue.append((current_prefix + (name,), child))
                    continue

                queue.append((current_prefix, child))

        return tuple(results)

    def _insert(self, sequence: NGramType) -> None:
        """
        Inserts a token in PrefixTrie

        Args:
            sequence (NGramType): Tokens to insert.
        """

        current_node = self._root

        for token in sequence:
            children_with_token = current_node.get_children(token)
            if children_with_token:
                current_node = children_with_token[0]
            else:
                current_node.add_child(token)
                current_node = current_node.get_children(token)[0]

class NGramTrieLanguageModel(PrefixTrie, NGramLanguageModel):
    """
    Trie specialized for storing and updating n-grams with frequency information.
    """

    #: N-gram window size used for building the trie
    _n_gram_size: int

    def __init__(self, encoded_corpus: tuple | None, n_gram_size: int) -> None:
        """
        Initialize an NGramTrieLanguageModel.

        Args:
            encoded_corpus (tuple | None): Encoded text
            n_gram_size (int): A size of n-grams to use for language modelling
        """
        NGramLanguageModel.__init__(self, encoded_corpus, n_gram_size)
        self._root = TrieNode()

    def __str__(self) -> str:
        """
        Return a string representation of the NGramTrieLanguageModel.

        Returns:
            str: String representation showing n-gram size.
        """
        return f"NGramTrieLanguageModel(n_gram_size={self._n_gram_size})"

    def build(self) -> int:
        """
        Build the trie using sliding n-gram windows from a tokenized corpus.

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1
        """

        self.clean()

        if not isinstance(self._encoded_corpus, tuple) or not self._encoded_corpus:
            return 1

        for sequence in self._encoded_corpus:
            if not isinstance(sequence, tuple) or len(sequence) < self._n_gram_size:
                continue

            for index in range(len(sequence) - self._n_gram_size + 1):
                ngram = tuple(sequence[index : index + self._n_gram_size])
                self._insert(ngram)

        ngrams = self._collect_all_ngrams()
        self._fill_frequencies(ngrams)

        return 0

    def get_next_tokens(self, start_sequence: NGramType) -> dict[int, float]:
        """
        Get all possible next tokens and their relative frequencies for a given prefix.

        Args:
            start_sequence (NGramType): The prefix sequence.

        Returns:
            dict[int, float]: Mapping of token → relative frequency.
        """
        node = self.get_prefix(start_sequence)

        if not node.has_children():
            return {}

        return self._collect_frequencies(node)

    def get_root(self) -> TrieNode:
        """
        Get the root.
        Returns:
            TrieNode: Found root.
        """

        return self._root

    def generate_next_token(self, sequence: NGramType) -> dict[int, float] | None:
        """
        Retrieve tokens that can continue the given sequence along with their probabilities.

        Uses the last (n_gram_size - 1) tokens as context to predict the next token.

        Args:
            sequence (NGramType): A sequence to match beginning of NGrams for continuation

        Returns:
            dict[int, float] | None: Possible next tokens with their probabilities,
                                     or None if input is invalid or context is too short
        """
        if not isinstance(sequence, tuple) or not sequence:
            return None

        if len(sequence) < self._n_gram_size - 1:
            return None

        context = sequence[-(self._n_gram_size - 1):]
        try:
            return self.get_next_tokens(context)
        except TriePrefixNotFoundError:
            return {}

    def get_n_gram_size(self) -> int:
        """
        Get the configured n-gram size.

        Returns:
            int: The current n-gram size.
        """
        return self._n_gram_size

    def get_node_by_prefix(self, prefix: NGramType) -> TrieNode:
        """
        Get the node corresponding to a prefix in the trie.

        Args:
            prefix (NGramType): Prefix to find node by.

        Returns:
            TrieNode: Found node by prefix.
        """
        return self.get_prefix(prefix)

    def update(self, new_corpus: tuple[NGramType]) -> None:
        """
        Update the trie with additional data and refresh frequency values.

        Args:
            new_corpus (tuple[NGramType]): Additional corpus represented as token sequences.
        """
        if not self._encoded_corpus:
            self._encoded_corpus = new_corpus
        else:
            self._encoded_corpus += new_corpus

        self.build()

    def _collect_all_ngrams(self) -> tuple[NGramType, ...]:
        """
        Collect all n-grams from the trie by traversing all paths of length n_gram_size.

        Returns:
            tuple[NGramType, ...]: Tuple of all n-grams stored in the trie.
        """
        result = []
        stack = [(self._root, [], 0)]
        while stack:
            node, current_path, current_depth = stack.pop()
            if node.get_name() is not None:
                current_path = current_path + [node.get_name()]
                current_depth += 1
            if current_depth == self._n_gram_size:
                result.append(tuple(current_path))
                continue
            children = node.get_children()
            for child in children:
                stack.append((child, current_path.copy(), current_depth))
        return tuple(result)

    def _collect_frequencies(self, node: TrieNode) -> dict[int, float]:
        """
        Collect frequencies from immediate child nodes only.

        Args:
            node (TrieNode): Current node.

        Returns:
            dict[int, float]: Collected frequencies of items.
        """
        frequencies = {}
        children = node.get_children()
        for child in children:
            token = child.get_name()
            if token is not None:
                frequencies[token] = child.get_value()
        return frequencies

    def _fill_frequencies(self, encoded_corpus: tuple[NGramType, ...]) -> None:
        """
        Calculate and assign frequencies for nodes in the trie based on corpus statistics.

        Counts occurrences of each n-gram and stores the relative frequency on the last node
        of each n-gram sequence.

        Args:
            encoded_corpus (tuple[NGramType, ...]): Tuple of n-grams extracted from the corpus.
        """
        if not encoded_corpus:
            return

        ngram_abs_frequency = {}

        for ngram in encoded_corpus:
            ngram_abs_frequency[ngram] = ngram_abs_frequency.get(ngram, 0) + 1

        len_corpus = len(encoded_corpus)

        for ngram, absolute_frequency in ngram_abs_frequency.items():
            relative_frequency = absolute_frequency / len_corpus
            last_node = self.get_prefix(ngram)
            last_node.set_value(relative_frequency)


class DynamicNgramLMTrie(NGramTrieLanguageModel):
    """
    Trie specialized in storing all possible N-grams tries.
    """

    #: Initial state of the tree
    _root: TrieNode
    #: Current size of ngrams
    _current_n_gram_size: int
    #: Maximum ngram size
    _max_ngram_size: int
    #: Models for text generation
    _models: dict[int, NGramTrieLanguageModel]
    #: Encoded corpus to generate text
    _encoded_corpus: tuple[NGramType, ...]

    def __init__(self, encoded_corpus: tuple[NGramType, ...], n_gram_size: int = 3) -> None:
        """
        Initialize an DynamicNgramLMTrie.

        Args:
            encoded_corpus (tuple[NGramType, ...]): Tokenized corpus.
            n_gram_size (int, optional): N-gram size. Defaults to 3.
        """
        super().__init__(encoded_corpus, n_gram_size)
        self._current_n_gram_size = 0
        self._max_ngram_size = n_gram_size
        self._models = {}

    def build(self) -> int:
        """
        Build N-gram tries for all possible ngrams based on a corpus of tokens.

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1.
        """
        if not isinstance(self._encoded_corpus, tuple) or not self._encoded_corpus:
            return 1

        if not isinstance(self._max_ngram_size, int) or self._max_ngram_size < 2:
            return 1

        for ngram_size in range(2, self._max_ngram_size + 1):
            model = NGramTrieLanguageModel(self._encoded_corpus, ngram_size)
            model.build()
            self._models[ngram_size] = model

        try:
            self._merge()
        except MergeTreesError:
            return 1

        return 0

    def set_current_ngram_size(self, current_n_gram_size: int | None) -> None:
        """
        Set the active N-gram size used for generation.

        Args:
            current_n_gram_size (int | None): Current N-gram size for generation.
        """
        if (
            not isinstance(current_n_gram_size, int)
            or not (2 <= current_n_gram_size <= self._max_ngram_size)
         ):
            raise IncorrectNgramError

        self._current_n_gram_size = current_n_gram_size

    def generate_next_token(self, sequence: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve tokens that can continue the given sequence along with their probabilities.

        Args:
            sequence (tuple[int, ...]): A sequence to match beginning of N-grams for continuation.

        Returns:
            dict[int, float] | None: Possible next tokens with their probabilities.
        """
        if not isinstance(sequence, tuple):
            return None

        if len(sequence) == 0:
            return None

        context = tuple()
        if len(sequence) < self._current_n_gram_size:
            context = sequence
        else:
            context = sequence[-(self._current_n_gram_size-1):]

        current_node = self._root

        for token in context:
            child: tuple[TrieNode, ...] = current_node.get_children(token)

            if not child:
                return {}

            current_node = child[0]

        candidates: dict[int, float] = {}
        for node in current_node.get_children():
            name = node.get_name()
            if name is None:
                continue
            candidates[name] = node.get_value()

        return candidates

    def _assign_child(self, parent: TrieNode, node_name: int, freq: float = 0.0) -> TrieNode:
        """
        Return an existing child with name of node or create a new one.

        Args:
            parent (TrieNode): A sequence to match beginning of N-grams for continuation.
            node_name (int): Name of TrieNode to find a child.
            freq (float, optional): Frequency of child TrieNode.

        Returns:
            TrieNode: Existing or new TrieNode.
        """
        if not isinstance(node_name, int) or node_name < 0:
            raise ValueError

        if not parent.get_children(node_name):
            parent.add_child(node_name)

        child = parent.get_children(node_name)[0]

        if freq != 0.0:
            child.set_value(child.get_value() + freq)

        return child

    def _merge(self) -> None:
        """
        Merge all built N-gram trie models into a single unified trie.
        """
        if not self._models:
            raise MergeTreesError

        self.clean()

        for trie in self._models.values():
            self._insert_trie(trie.get_root())

    def _insert_trie(self, source_root: TrieNode) -> None:
        """
        Insert all nodes of source root trie into our main root.

        Args:
            source_root (TrieNode): Source root to insert tree
        """
        queue = [(source_root, self._root)]

        while queue:
            node, parent = queue.pop(0)

            for child in node.get_children():
                name = child.get_name()
                if name is None:
                    continue
                value = child.get_value()
                new_parrent = self._assign_child(parent, name, value)
                queue.append((child, new_parrent))


class DynamicBackOffGenerator(BackOffGenerator):
    """
    Dynamic back-off generator based on dynamic N-gram trie.
    """

    #: Dynamic trie for text generation
    _dynamic_trie: DynamicNgramLMTrie

    def __init__(self, dynamic_trie: DynamicNgramLMTrie, processor: WordProcessor) -> None:
        """
        Initialize an DynamicNgramLMTrie.

        Args:
            dynamic_trie (DynamicNgramLMTrie): Dynamic trie to use for text generation.
            processor (WordProcessor): A WordProcessor instance to handle text processing.
        """
        super().__init__((dynamic_trie, ), processor)
        self._dynamic_trie = dynamic_trie

    def get_next_token(self, sequence_to_continue: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            dict[int, float] | None: Next tokens for sequence continuation
        """
        if not isinstance(sequence_to_continue, tuple) or not sequence_to_continue:
            return None

        for ngram_size in range(self._dynamic_trie.get_n_gram_size(), 1, -1):

            context = sequence_to_continue[-min(ngram_size - 1, len(sequence_to_continue)):]

            self._dynamic_trie.set_current_ngram_size(ngram_size)

            candidates = self._dynamic_trie.generate_next_token(context)

            if candidates:
                return candidates

        return None

    def run(self, seq_len: int, prompt: str) -> str | None:
        """
        Generate sequence based on dynamic N-gram trie and prompt provided.

        Args:
            seq_len (int): Number of tokens to generate
            prompt (str): Beginning of sequence

        Returns:
            str | None: Generated sequence
        """
        if not isinstance(prompt, str) or not check_positive_int(seq_len):
            return None

        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None

        result_sequence = list(encoded_prompt)

        for _ in range(seq_len):
            candidates = self.get_next_token(tuple(result_sequence))
            if not candidates:
                break
            next_element = max(candidates.items(), key=lambda x: (x[1], x[0]))[0]
            result_sequence.append(next_element)

        decoded = tuple(
            token for token in (self._text_processor.get_token(t) for t in result_sequence)
            if token is not None
        )

        if not decoded:
            return None

        decoded_text = []
        sentence = []
        for token in decoded:
            if token == self._text_processor.get_end_of_word_token():
                decoded_text.append(" ".join(sentence).capitalize() + ".")
                sentence.clear()
                continue
            sentence.append(token)

        if sentence:
            decoded_text.append(" ".join(sentence).capitalize() + ".")

        return " ".join(decoded_text)

def save(trie: DynamicNgramLMTrie, path: str) -> None:
    """
    Save DynamicNgramLMTrie.

    Args:
        trie (DynamicNgramLMTrie): Trie for saving
        path (str): Path for saving
    """
    root = trie.get_root()

    root_json = {
        "value": root.get_name(),
        "freq": root.get_value(),
        "children": []
    }

    queue = [(root, root_json)]

    while queue:

        current_node, current_json = queue.pop()
        for child in current_node.get_children():
            child_json = {
                "value": child.get_name(),
                "freq": child.get_value(),
                "children": []
            }

            current_json["children"].append(child_json)
            queue.append((child, child_json))

    output_json = {"trie": root_json}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output_json, f)

def load(path: str) -> DynamicNgramLMTrie:
    """
    Load DynamicNgramLMTrie from file.

    Args:
        path (str): Trie path

    Returns:
        DynamicNgramLMTrie: Trie from file.
    """
    with open(path, "r", encoding="utf-8") as f:
        input_json = json.load(f)

    trie_json = input_json.get("trie")
    trie = DynamicNgramLMTrie(tuple())

    root = trie.get_root()

    queue = [(root, trie_json)]

    while queue:
        current_node, current_json = queue.pop(0)

        for child_json in current_json.get("children", []):
            value = child_json["value"]
            freq = child_json["freq"]

            children = current_node.get_children(value)

            child_node = TrieNode()
            if children:
                child_node = children[0]
            else:
                current_node.add_child(value)
                child_node = current_node.get_children(value)[0]

            child_node.set_value(freq)

            queue.append((child_node, child_json))

    return trie
