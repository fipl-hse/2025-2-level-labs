"""
Lab 4
"""

# pylint: disable=unused-argument, super-init-not-called, unused-private-member, duplicate-code, unused-import
import json

from lab_3_generate_by_ngrams.main import BackOffGenerator, NGramLanguageModel, TextProcessor

NGramType = tuple[int, ...]
"Type alias for NGram."

class CustomException(Exception):
    """Exception raised when something fails due to something"""


class TriePrefixNotFoundError(CustomException):
    """Raised when required prefix is not found in the trie"""


class EncodingError(CustomException):
    """Raised when text encoding fails due to incorrect input or processing error"""


class DecodingError(CustomException):
    """Raised when text decoding fails due to incorrect input or processing error"""


class IncorrectNgramError(CustomException):
    """Raised when attempting to use inappropriate n-gram size"""


class MergeTreesError(CustomException):
    """Raised when tree merging is impossible"""

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
        if not isinstance(text, str) or not text.strip():
            raise EncodingError("Invalid input: text must be a non-empty string")
        tokens = self._tokenize(text)
        encoded_sentences = []
        current_sentence = []
        for token in tokens:
            self._put(token)
            current_sentence.append(self._storage[token])
            if token == self._end_of_sentence_token:
                if current_sentence:
                    encoded_sentences.append(tuple(current_sentence))
                current_sentence = []
        if current_sentence:
            encoded_sentences.append(tuple(current_sentence))
        return tuple(encoded_sentences)

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
            new_id = len(self._storage)
            self._storage[element] = new_id

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
        sentences = []
        current_sentence = []
        for word in decoded_corpus:
            if word == self._end_of_sentence_token:
                if current_sentence:
                    sentence_str = ' '.join(current_sentence)
                    if sentence_str:
                        sentences.append(sentence_str.capitalize())
                    current_sentence = []
            else:
                current_sentence.append(word)
        if current_sentence:
            sentence_str = ' '.join(current_sentence)
            if sentence_str:
                sentences.append(sentence_str.capitalize())
        if not sentences:
            raise DecodingError("Postprocessing resulted in empty output")
        result = '. '.join(sentences)
        if result and result[-1] != '.':
            result += '.'
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
        if not isinstance(text, str) or not text.strip():
            raise EncodingError("Invalid input: text must be a non-empty string")
        tokens = []
        sentences = []
        current_sentence = []
        for char in text:
            if char in '.!?':
                sentence = ''.join(current_sentence).strip()
                if sentence:
                    sentences.append(sentence)
                current_sentence = []
            else:
                current_sentence.append(char)
        last_sentence = ''.join(current_sentence).strip()
        if last_sentence:
            sentences.append(last_sentence)
        for sentence in sentences:
            words = sentence.split()
            cleaned_words = []
            for word in words:
                word = word.lower()
                cleaned_word = ''.join(char for char in word if char.isalpha())
                if cleaned_word:
                    cleaned_words.append(cleaned_word)
            if cleaned_words:
                tokens.extend(cleaned_words)
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
        return len(self._children) > 0

    def __str__(self) -> str:
        """
        Return a string representation of the N-gram node.

        Returns:
            str: String representation showing node data and frequency.
        """
        return f"TrieNode(name={self.__name}, value={self._value})"

    def add_child(self, item: int) -> None:
        """
        Add a new child node with the given item.

        Args:
            item (int): Data value for the new child node.
        """
        new_node = TrieNode(item)
        self._children.append(new_node)

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
        children_with_item = tuple(child for child in self._children if child.get_name() == item)
        return children_with_item

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
            found = False
            for child in current_node.get_children():
                if child.get_name() == item:
                    current_node = child
                    found = True
                    break
            if not found:
                raise TriePrefixNotFoundError(f"Prefix {prefix} not found in trie")
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
            prefix_node = self.get_prefix(prefix)
        except TriePrefixNotFoundError:
            return tuple()
        sequences = []
        prefix_list = [item for item in prefix if item is not None]
        stack = [(prefix_node, prefix_list)]
        while stack:
            current_node, current_sequence = stack.pop()
            children = []
            for child in current_node.get_children():
                if child.get_name() is not None:
                    children.append(child)
            for i in range(len(children) - 1, -1, -1):
                child = children[i]
                child_name = child.get_name()
                if child_name is None:
                    continue
                new_sequence = current_sequence + [child_name]
                if not child.has_children():
                    sequences.append(tuple(new_sequence))
                else:
                    stack.append((child, new_sequence))
        return tuple(sequences)

    def _insert(self, sequence: NGramType) -> None:
        """
        Inserts a token in PrefixTrie

        Args:
            sequence (NGramType): Tokens to insert.
        """
        current_node = self._root
        for item in sequence:
            existing_child = None
            for child in current_node.get_children():
                if child.get_name() == item:
                    existing_child = child
                    break
            if existing_child is None:
                current_node.add_child(item)
                for child in current_node.get_children():
                    if child.get_name() == item:
                        current_node = child
                        break
            else:
                current_node = existing_child


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
        self._root = TrieNode()
        all_ngrams = []
        for sentence in self._encoded_corpus:
            if len(sentence) < self._n_gram_size:
                continue
            for i in range(len(sentence) - self._n_gram_size + 1):
                ngram = tuple(sentence[i:i + self._n_gram_size])
                all_ngrams.append(ngram)
                self._insert(ngram)
        collected_ngrams = self._collect_all_ngrams()
        self._fill_frequencies(collected_ngrams)
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
        if self._encoded_corpus is None:
            self._encoded_corpus = new_corpus
        else:
            self._encoded_corpus = tuple(list(self._encoded_corpus) + list(new_corpus))
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
                ngrams.append(tuple(current_path))
                continue
            for child in reversed(current_node.get_children()):
                if child.get_name() is not None:
                    stack.append((child, current_path + [child.get_name()]))
        return tuple(ngrams)

    def _collect_frequencies(self, node: TrieNode) -> dict[int, float]:
        """
        Collect frequencies from immediate child nodes only.

        Args:
            node (TrieNode): Current node.

        Returns:
            dict[int, float]: Collected frequencies of items.
        """
        frequencies = {}
        for child in node.get_children():
            name = child.get_name()
            if name is not None:
                frequencies[name] = child.get_value()
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
        ngram_counts = {}
        for ngram in encoded_corpus:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        total_ngrams = len(encoded_corpus)
        for ngram, count in ngram_counts.items():
            relative_frequency = count / total_ngrams
            try:
                node = self.get_prefix(ngram)
                node.set_value(relative_frequency)
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
        self._root = TrieNode()
        self._current_n_gram_size = 0
        self._max_ngram_size = n_gram_size
        self._models = {}
        self._encoded_corpus = encoded_corpus
        self._n_gram_size = n_gram_size

    def build(self) -> int:
        """
        Build N-gram tries for all possible ngrams based on a corpus of tokens.

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1.
        """
        if not isinstance(self._max_ngram_size, int) or self._max_ngram_size < 2:
            return 1
        if not isinstance(self._encoded_corpus, tuple):
            return 1
        if not self._encoded_corpus:
            return 1
        for sentence in self._encoded_corpus:
            if not isinstance(sentence, tuple):
                return 1
            if not all(isinstance(x, int) for x in sentence):
                return 1
        for ngram_size in range(2, self._max_ngram_size + 1):
            model = NGramTrieLanguageModel(self._encoded_corpus, ngram_size)
            build_result = model.build()
            if build_result == 0:
                self._models[ngram_size] = model
            else:
                return 1
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
        if current_n_gram_size is None:
            self._current_n_gram_size = self._max_ngram_size
            return
        if not isinstance(current_n_gram_size, int) or current_n_gram_size < 2:
            raise IncorrectNgramError(f"N-gram size must be between 2 and {self._max_ngram_size}")
        if current_n_gram_size > self._max_ngram_size:
            raise IncorrectNgramError(f"N-gram size must be between 2 and {self._max_ngram_size}")
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
        context_size = min(self._current_n_gram_size - 1, len(sequence))
        if context_size <= 0:
            return {}
        context = sequence[-context_size:]
        try:
            current_node = self._root
            for item in context:
                found = False
                for child in current_node.get_children():
                    if child.get_name() == item:
                        current_node = child
                        found = True
                        break
                if not found:
                    return {}
            frequencies = dict[int, float]()
            for child in current_node.get_children():
                name = child.get_name()
                if name is not None and child.get_value() > 0:
                    frequencies[name] = child.get_value()
            return frequencies
        except TriePrefixNotFoundError:
            return {}

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
            raise ValueError("Node name must be a non-negative integer")
        for child in parent.get_children():
            if child.get_name() == node_name:
                if freq > 0:
                    child.set_value(freq)
                return child
        new_child = TrieNode(node_name, freq)
        if hasattr(parent, '_children'):
            parent._children.append(new_child)
        return new_child

    def _merge(self) -> None:
        """
        Merge all built N-gram trie models into a single unified trie.
        """
        if not self._models:
            raise MergeTreesError("No models to merge")
        self._root = TrieNode()
        for ngram_size in sorted(self._models.keys()):
            model = self._models[ngram_size]
            self._insert_trie(model.get_root())

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
            for source_child in source_node.get_children():
                if source_child.get_name() is not None:
                    source_child_name = source_child.get_name()
                if source_child_name is not None:
                    target_child = self._assign_child(
                        target_node,
                        source_child_name,
                        source_child.get_value()
                    )
                    if source_child.has_children():
                        stack.append((source_child, target_child))


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
        max_possible_size = min(self._dynamic_trie._max_ngram_size, len(sequence_to_continue) + 1)
        ngram_sizes = list(range(max_possible_size, 1, -1))
        for ngram_size in ngram_sizes:
            try:
                self._dynamic_trie.set_current_ngram_size(ngram_size)
                next_tokens = self._dynamic_trie.generate_next_token(sequence_to_continue)
                if next_tokens:
                    return next_tokens
            except (IncorrectNgramError, TriePrefixNotFoundError):
                continue
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
        if not isinstance(seq_len, int) or seq_len <= 0:
            return None
        if not isinstance(prompt, str) or not prompt.strip():
            return None
        encoded_prompt_test = self._text_processor.encode(prompt)
        if encoded_prompt_test is None:
             return None
        try:
            tokens = list(self._text_processor._tokenize(prompt))
        except EncodingError:
            return None
        eos_token = self._text_processor._end_of_sentence_token
        if tokens and tokens[-1] == eos_token:
            tokens.pop()
        encoded_prompt = []
        for token in tokens:
            self._text_processor._put(token)
            encoded_prompt.append(self._text_processor._storage[token])
        if not encoded_prompt:
            return None
        current_sequence = list(encoded_prompt)
        for _ in range(seq_len):
            next_tokens = self.get_next_token(tuple(current_sequence))
            if not next_tokens:
                break
            best_token = sorted(next_tokens.items(), key=lambda x: (-x[1], -x[0]))[0][0]
            current_sequence.append(best_token)
        decoded_words = []
        for token_id in current_sequence:
            for word, word_id in self._text_processor._storage.items():
                if word_id == token_id:
                    decoded_words.append(word)
                    break
        postprocess_method = getattr(self._text_processor, '_postprocess_decoded_text', None)
        if postprocess_method:
            return str(postprocess_method(tuple(decoded_words)))
        return ' '.join(decoded_words)

def save(trie: DynamicNgramLMTrie, path: str) -> None:
    """
    Save DynamicNgramLMTrie.

    Args:
        trie (DynamicNgramLMTrie): Trie for saving
        path (str): Path for saving
    """
    stack = [(trie.get_root(), {} if False else None)]
    root_dict = None
    while stack:
        current_node, parent_dict = stack.pop()
        node_dict = {
            "value": current_node.get_name(),
            "freq": current_node.get_value(),
            "children": []
        }
        if parent_dict is None:
            root_dict = node_dict
        else:
            parent_dict["children"].append(node_dict)
        for child in reversed(current_node.get_children()):
            stack.append((child, node_dict))
    trie_data = {
        "trie": root_dict
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(trie_data, f, indent=2)

def load(path: str) -> DynamicNgramLMTrie:
    """
    Load DynamicNgramLMTrie from file.

    Args:
        path (str): Trie path

    Returns:
        DynamicNgramLMTrie: Trie from file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        trie_data = json.load(f)
    trie = DynamicNgramLMTrie((), 3)
    root_dict = trie_data.get("trie")
    if root_dict:
        root_value = root_dict.get("value")
        if root_value is not None:
            trie._root = TrieNode(root_value, root_dict.get("freq", 0.0))
        else:
            trie._root = TrieNode(None, root_dict.get("freq", 0.0))
        stack = [(root_dict, trie._root)]
        while stack:
            current_dict, current_node = stack.pop()
            for child_dict in reversed(current_dict.get("children", [])):
                child_node = TrieNode(child_dict.get("value"), child_dict.get("freq", 0.0))
                current_node._children.append(child_node)
                stack.append((child_dict, child_node))
    return trie
