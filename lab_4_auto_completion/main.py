"""
Lab 4
"""

# pylint: disable=unused-argument, super-init-not-called, unused-private-member, duplicate-code, unused-import
import json

from lab_3_generate_by_ngrams.main import BackOffGenerator, NGramLanguageModel, TextProcessor

NGramType = tuple[int, ...]
"Type alias for NGram."


class TriePrefixNotFoundError(Exception):
    """Exception raised when required prefix is not found in the trie."""


class EncodingError(Exception):
    """Exception raised when text encoding fails due to incorrect input or processing error."""


class DecodingError(Exception):
    """Exception raised when text decoding fails due to incorrect input or processing error."""


class IncorrectNgramError(Exception):
    """Exception raised when trying to use inappropriate n-gram size."""


class MergeTreesError(Exception):
    """Exception raised when it's impossible to merge trees."""


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
        self._end_of_sentence_token = end_of_sentence_token
        super().__init__(end_of_word_token=end_of_sentence_token)

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
        if not isinstance(text, str) or not text:
            raise EncodingError("Input text must be a non-empty string")

        tokens = self._tokenize(text)
        sentences = []
        current_sentence = []

        for token in tokens:
            current_sentence.append(token)
            if token == self._end_of_sentence_token:
                encoded_sentence = []
                for word in current_sentence:
                    self._put(word)
                    word_id = self.get_id(word)
                    encoded_sentence.append(word_id)

                sentences.append(tuple(encoded_sentence))
                current_sentence = []

        return tuple(sentences)

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): An element to put into storage

        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """
        if not isinstance(element, str):
            return

        if element not in self._storage:
            self._storage[element] = len(self._storage)

        return None

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
        if all(token == self._end_of_sentence_token for token in decoded_corpus):
            raise DecodingError("Postprocessing resulted in empty output")
        sentences = []
        current_sentence = []
        for word in decoded_corpus:
            if word == self._end_of_sentence_token:
                if current_sentence:
                    sentences.append(" ".join(current_sentence))
                    current_sentence = []
            else:
                current_sentence.append(word)
        if current_sentence:
            sentences.append(" ".join(current_sentence))
        if not sentences:
            raise DecodingError("Postprocessing resulted in empty output")
        processed_sentences = []
        for sentence in sentences:
            if sentence.strip():
                sentence = sentence.strip()
                capitalized = sentence[0].upper() + sentence[1:] if sentence else ""
                processed_sentences.append(capitalized + ".")
        result = " ".join(processed_sentences)
        return result

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
        sentences = []
        current = ""

        for char in text:
            current += char
            if char in ".!?":
                sentences.append(current)
                current = ""

        if current:
            sentences.append(current)

        for sentence in sentences:
            if not sentence.strip():
                continue

            words = sentence.lower().split()

            for word in words:
                cleaned_word = "".join(filter(str.isalpha, word))
                if cleaned_word:
                    tokens.append(cleaned_word)

            tokens.append(self._end_of_sentence_token)

        if not tokens:
            raise EncodingError("No valid words found after tokenization")

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
        self.name = name
        self.value = value
        self._children = []
        self.is_end = False
        """
        Initialize a Trie node.

        Args:
            name (int | None, optional): The name of the node.
            value (float, optional): The value stored in the node.
        """
        self._name = name
        self._value = value
        self._children = []
        self.is_end = False

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
        return f"TrieNode(name={self.name}, value={self.value})"

    def add_child(self, item: int) -> None:
        """
        Add a new child node with the given item.

        Args:
            item (int): Data value for the new child node.
        """
        child_node = TrieNode(name=item, value=0.0)
        self._children.append(child_node)

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

        return tuple(child for child in self._children if child.get_name() == item)

    def get_name(self) -> int | None:
        """
        Get the data stored in the node.

        Returns:
            int | None: TrieNode data.
        """
        return self.name

    def get_value(self) -> float:
        """
        Get the value of the node.

        Returns:
            float: Frequency value.
        """
        return self.value

    def set_value(self, new_value: float) -> None:
        """
        Set the value of the node

        Args:
            new_value (float): New value to store.
        """
        self.value = new_value

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
        self._root = TrieNode(name=None, value=0.0)

    def clean(self) -> None:
        self._root = TrieNode()
        """
        Clean the whole tree.
        """
        self.root = TrieNode()

    def fill(self, encoded_corpus: tuple[NGramType]) -> None:
        """
        Fill the trie based on an encoded_corpus of tokens.

        Args:
            encoded_corpus (tuple[NGramType]): Tokenized corpus.
        """
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
            children_tuple = current_node.get_children(item)

            if len(children_tuple) != 1:
                raise TriePrefixNotFoundError()

            current_node = children_tuple[0]

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

        suggestions = []

        if start_node.is_end:
            suggestions.append(tuple(prefix))

        stack = [(start_node, list(prefix))]

        while stack:
            current_node, current_path = stack.pop()

            children = current_node.get_children()
            sorted_children = sorted(
                [child for child in children if child.get_name() is not None],
                key=lambda child: child.get_name()
            )

            for child in sorted_children:
                item = child.get_name()
                if item is not None:
                    new_path = current_path + [item]
                    
                    if child.is_end:
                        suggestions.append(tuple(new_path))

                if child.has_children():
                    stack.append((child, new_path))

        return tuple(sorted(suggestions))

    def _insert(self, sequence: NGramType) -> None:
        """
        Inserts a token in PrefixTrie

        Args:
            sequence (NGramType): Tokens to insert.
        """
        current_node = self._root

        for item in sequence:
            child_node = None
            children = current_node.get_children()

            for child in children:
                if child.get_name() == item:
                    child_node = child
                    break

            if child_node is None:
                current_node.add_child(item)
                child_node = current_node.get_children(item)[0]

            current_node = child_node

        current_node.is_end = True


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
        self._n_gram_size = n_gram_size
        self._max_ngram_size = n_gram_size

    def __str__(self) -> str:
        """
        Return a string representation of the NGramTrieLanguageModel.

        Returns:
            str: String representation showing n-gram size.
        """
        return f"NGramTrieLanguageModel({self._n_gram_size})"

    def build(self) -> int:
        """
        Build the trie using sliding n-gram windows from a tokenized corpus.

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1
        """
        if not self._encoded_corpus:
            return 1
        ngrams = []
        for sentence in self._encoded_corpus:
            for i in range(len(sentence) - self._n_gram_size + 1):
                ngram = sentence[i : i + self._n_gram_size]
                ngrams.append(ngram)
        try:
            for ngram in ngrams:
                self._insert(ngram)
            all_ngrams = self._collect_all_ngrams()
            self._fill_frequencies(all_ngrams)
            return 0
        except (TriePrefixNotFoundError, EncodingError, DecodingError):
            return 1

    def get_next_tokens(self, start_sequence: NGramType) -> dict[int, float]:
        """
        Get all possible next tokens and their relative frequencies for a given prefix.

        Args:
            start_sequence (NGramType): The prefix sequence.

        Returns:
            dict[int, float]: Mapping of token → relative frequency.
        """
        if not isinstance(start_sequence, tuple) or not start_sequence:
            return {}

        if len(start_sequence) < self._n_gram_size - 1:
            return {}

        context = start_sequence[-(self._n_gram_size - 1) :]

        node = self.get_node_by_prefix(context)
        result = {}

        for child in node.get_children():
            token = child.get_name()
            if token is not None:
                result[token] = child.get_value()

        return result

        return {}

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
        if not isinstance(sequence, tuple):
            return None

        if not sequence:
            return None

        if len(sequence) < self._n_gram_size - 1:
            return None

        try:
            result = self.get_next_tokens(sequence)
            return result

        except TriePrefixNotFoundError:
            return {}

    def get_n_gram_size(self) -> int:
        return self._n_gram_size
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
        if not new_corpus:
            return

        if self._encoded_corpus is None:
            self._encoded_corpus = new_corpus
        else:
            self._encoded_corpus = self._encoded_corpus + new_corpus

        self.build()

    def _collect_all_ngrams(self) -> tuple[NGramType, ...]:
        """
        Collect all n-grams from the trie by traversing all paths of length n_gram_size.

        Returns:
            tuple[NGramType, ...]: Tuple of all n-grams stored in the trie.
        """
        ngrams = []
        stack = [(self._root, [])]

        while stack:
            current_node, current_path = stack.pop()

            if len(current_path) == self._n_gram_size:
                if current_node.is_end:
                    ngrams.append(tuple(current_path))
                continue

            children = current_node.get_children()
            for child in children:
                token = child.get_name()
                if token is not None:
                    stack.append((child, current_path + [token]))

        if not ngrams:
            return False
        return tuple(ngrams)

    def _collect_frequencies(self, node: TrieNode) -> dict[int, float]:
        """
        Collect frequencies from immediate child nodes only.

        Args:
            node (TrieNode): Current node.

        Returns:
            dict[int, float]: Collected frequencies of items.
        """
        result = {}
        if node:
            for child in node.get_children():
                token = child.get_name()
                if token is not None:
                    result[token] = child.get_value()
        return result

    def _fill_frequencies(self, encoded_corpus: tuple[NGramType, ...]) -> None:
        """
        Calculate and assign frequencies for nodes in the trie based on corpus statistics.

        Counts occurrences of each n-gram and stores the relative frequency on the last node
        of each n-gram sequence.

        Args:
            encoded_corpus (tuple[NGramType, ...]): Tuple of n-grams extracted from the corpus.
        """
        absolute_counts = {}
        for ngram in encoded_corpus:
            absolute_counts[ngram] = absolute_counts.get(ngram, 0) + 1

        total_ngrams = len(encoded_corpus)

        for ngram, count in absolute_counts.items():
            relative_frequency = count / total_ngrams

            try:
                last_node = self.get_prefix(ngram)
                last_node.set_value(relative_frequency)
            except TriePrefixNotFoundError:
                continue


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
        self._root = TrieNode()
        self._encoded_corpus = encoded_corpus
        self._current_n_gram_size = 0
        self._max_ngram_size = n_gram_size
        self._models = {}
        self._n_gram_size = n_gram_size

    def build(self) -> int:
        """
        Build N-gram tries for all possible ngrams based on a corpus of tokens.

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1.
        """
        if (
            not isinstance(self._encoded_corpus, tuple)
            or not self._encoded_corpus
            or not isinstance(self._max_ngram_size, int)
            or self._max_ngram_size < 2
        ):
            return 1
        self._models = {}
        for ngram_size in range(2, self._max_ngram_size + 1):
            model = NGramTrieLanguageModel(self._encoded_corpus, ngram_size)
            if model.build() != 0:
                return 1
            self._models[ngram_size] = model
        try:
            self._merge()
            return 0
        except MergeTreesError:
            return 1

    def set_current_ngram_size(self, current_n_gram_size: int | None) -> None:
        """
        Set the active N-gram size used for generation.

        Args:
            current_n_gram_size (int | None): Current N-gram size for generation.
        """
        if (
            not isinstance(current_n_gram_size, int)
            or current_n_gram_size is None
            or current_n_gram_size < 2
            or self._max_ngram_size < current_n_gram_size
        ):
            raise IncorrectNgramError("Invalid n-gram size")
        self._current_n_gram_size = current_n_gram_size

    def generate_next_token(self, sequence: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve tokens that can continue the given sequence along with their probabilities.

        Args:
            sequence (tuple[int, ...]): A sequence to match beginning of N-grams for continuation.

        Returns:
            dict[int, float] | None: Possible next tokens with their probabilities.
        """
        if not isinstance(sequence, tuple) or not sequence:
            return None
        if not 2 <= self._current_n_gram_size <= self._max_ngram_size:
            self._current_n_gram_size = self._max_ngram_size

        if self._current_n_gram_size not in self._models:
            found_size = None
            for candidate_size in range(self._max_ngram_size, 1, -1):
                if candidate_size in self._models:
                    found_size = candidate_size
                    break

            if found_size is None:
                return {}

            self._current_n_gram_size = found_size

        selected_model = self._models[self._current_n_gram_size]
        current_ngram_size = selected_model.get_n_gram_size()

        if len(sequence) < current_ngram_size - 1:
            return {}

        context_length = min(current_ngram_size - 1, len(sequence))
        context_sequence = sequence[-context_length:]

        try:
            context_node = selected_model.get_node_by_prefix(context_sequence)

            next_tokens = {}
            child_nodes = context_node.get_children()

            for child_node in child_nodes:
                token_id = child_node.get_name()
                if token_id is not None:
                    token_frequency = child_node.get_value()
                    next_tokens[token_id] = token_frequency

            return next_tokens

        except TriePrefixNotFoundError:
            return {}

        current_node = self._dynamic_trie._root

        context_length = min(len(sequence), self._current_n_gram_size - 1)

        for i in range(context_length):
            children = current_node.get_children(sequence[i])
            if not children:
                return {}
            current_node = children[0]

        result = {}
        all_children = current_node.get_children()

        for child in all_children:
            child_name = child.get_name()
            child_freq = child.get_value()

            if child_name is not None and child_freq != 0.0:
                result[child_name] = child_freq

        return result if result else {}

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
        if node_name is None or not isinstance(node_name, int) or node_name < 0:
            raise ValueError("Node name must be non-negative integer")
        for child in parent.get_children():
            if child.get_name() == node_name:
                if freq != 0.0:
                    child.set_value(freq)
                return child
        parent.add_child(node_name)
        for child in parent.get_children():
            if child.get_name() == node_name:
                if freq != 0.0:
                    child.set_value(freq)
                return child
        return child

    def _merge(self) -> None:
        """
        Merge all built N-gram trie models into a single unified trie.
        """
        if not self._models:
            raise MergeTreesError()

        self._root = TrieNode()

        for n_size in sorted(self._models.keys()):
            model = self._models[n_size]
            source_root = model.get_root()
            self._insert_trie(source_root)

    def _insert_trie(self, source_root: TrieNode) -> None:
        """
        Insert all nodes of source root trie into our main root.

        Args:
            source_root (TrieNode): Source root to insert tree
        """
        if not source_root or not source_root.has_children():
            return

        stack = [(source_root, self._root)]

        while stack:
            source_node, target_node = stack.pop()

            children = source_node.get_children()
            for child in children:
                child_name = child.get_name()
                child_freq = child.get_value()

                if child_name is not None:
                    target_child = self._assign_child(target_node, child_name, child_freq)

                if child.has_children():
                    stack.append((child, target_child))

                if child.is_end:
                    target_child.is_end = True


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
        super().__init__((dynamic_trie,), processor)
        self._dynamic_trie = dynamic_trie
        self._processor = processor

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

        if len(sequence_to_continue) >= self._dynamic_trie._max_ngram_size:
            max_search_size = self._dynamic_trie._max_ngram_size
        else:
            max_search_size = len(sequence_to_continue)

        available_sizes = sorted(self._dynamic_trie._models.keys(), reverse=True)

        if not available_sizes:
            return None

        for n_size in available_sizes:
            if n_size > max_search_size + 1:
                continue

            self._dynamic_trie.set_current_ngram_size(n_size)

            next_tokens = self._dynamic_trie.generate_next_token(sequence_to_continue)

            if next_tokens is None:
                continue
            if next_tokens:
                return next_tokens

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
        if not isinstance(seq_len, int) or seq_len < 0:
            return None
        if not isinstance(prompt, str) or not prompt:
            return None

        encoded_prompt = self._processor.encode(prompt)
        if not encoded_prompt:
            return None

        current_sequence = list(encoded_prompt)

        for _ in range(seq_len):
            next_tokens = self.get_next_token(tuple(current_sequence))

            if not next_tokens:
                break

            best_token = max(next_tokens.items(), key=lambda x: x[1])[0]
            current_sequence.append(best_token)

        result = self._processor.decode(tuple(current_sequence))

        return result if result else None


def save(trie: DynamicNgramLMTrie, path: str) -> None:
    """
    Save DynamicNgramLMTrie.

    Args:
        trie (DynamicNgramLMTrie): Trie for saving
        path (str): Path for saving
    """
    if not isinstance(path, str) or not path:
        raise ValueError("Invalid path")
    data = {
        "value": trie.get_root().get_name(),
        "freq": trie.get_root().get_value(),
        "children": [],
    }
    stack = [(trie.get_root(), data["children"])]
    while stack:
        current_node, parent_children_list = stack.pop()
        children = current_node.get_children()
        for child in children:
            child_dict = {"value": child.get_name(), "freq": child.get_value(), "children": []}
            parent_children_list.append(child_dict)
            stack.append((child, child_dict["children"]))
    trie_data = {"trie": data}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(trie_data, file, indent=2)


def load(path: str) -> DynamicNgramLMTrie:
    """
    Load DynamicNgramLMTrie from file.

    Args:
        path (str): Trie path

    Returns:
        DynamicNgramLMTrie: Trie from file.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    encoded_corpus = tuple(data.get("encoded_corpus", ()))
    max_ngram_size = data.get("max_ngram_size", 3)
    loaded_trie = DynamicNgramLMTrie(encoded_corpus, max_ngram_size)
    result = loaded_trie.build()
    if result != 0:
        return DynamicNgramLMTrie(tuple(), max_ngram_size)
    loaded_trie.set_current_ngram_size(data.get("current_n_gram_size", max_ngram_size))
    return loaded_trie




