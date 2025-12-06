"""
Lab 4
"""

# pylint: disable=unused-argument, super-init-not-called, unused-private-member, duplicate-code, unused-import
import json

from lab_3_generate_by_ngrams.main import BackOffGenerator, NGramLanguageModel, TextProcessor

NGramType = tuple[int, ...]
"Type alias for NGram."

class TriePrefixNotFoundError(Exception):
    """
    Exception raised when the prefix required fot the transition is not in the tree.
    """

class EncodingError(Exception):
    """
    Exception raised when text encoding fails due to incorrect input or processing error.
    """

class DecodingError(Exception):
    """
    Exception raised when text decoding fails due to incorrect input or processing error.
    """

class IncorrectNgramError(Exception):
    """
    Exception raised when trying to use an inappropriate n-gram size.
    """

class MergeTreesError(Exception):
    """
    Exception raised when merging trees is not possible.
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
        super().__init__(end_of_word_token = end_of_sentence_token)
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
        raw_sentences = []
        current = []
        for char in text:
            current.append(char)
            if char in '.!?' and (len(current) == len(text) or text[len(current)].isspace()):
                sent = ''.join(current).strip()
                if sent:
                    raw_sentences.append(sent)
                current = []
        if current:
            sent = ''.join(current).strip()
            if sent:
                raw_sentences.append(sent)
        clean_sentences = [s.lower().strip() for s in raw_sentences]
        encoded = []
        for clean_sentence in clean_sentences:
            if not clean_sentence:
                continue
            tokens = self._tokenize(clean_sentence)
            encoded_sentence = []
            for token in tokens:
                self._put(token)
                word_id = self._storage[token]
                encoded_sentence.append(word_id)
            if encoded_sentence:
                if (self._end_of_sentence_token in self._storage and 
                    (not tokens or tokens[-1] != self._end_of_sentence_token)):
                    eos_id = self._storage[self._end_of_sentence_token]
                    encoded_sentence.append(eos_id)
                encoded.append(tuple(encoded_sentence))
        return tuple(encoded)

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
        if not isinstance(decoded_corpus, tuple) or len(decoded_corpus) == 0:
            raise DecodingError("Invalid input: decoded_corpus must be a non-empty tuple")
        sentences = []
        current_words = []
        for item in decoded_corpus:
            if item == self._end_of_sentence_token:
                if current_words:  # Если есть слова в текущем предложении
                    sentences.append(" ".join(current_words))
                    current_words = []
            else:
                current_words.append(item)
        if current_words:
            sentences.append(" ".join(current_words))
        if not sentences:
            raise DecodingError("Postprocessing resulted in empty output")
        result = ". ".join(sentence.capitalize() for sentence in sentences) + "."
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
        tokenized_words = []
        current_word = []
        for char in text:
            if char.isalpha(): 
                current_word.append(char.lower())
            else:
                if current_word:
                    word = ''.join(current_word)
                    tokenized_words.append(word)
                    current_word = []
                if char in '.!?':
                    if tokenized_words and tokenized_words[-1] != self._end_of_sentence_token:
                        tokenized_words.append(self._end_of_sentence_token)
        if current_word:
            word = ''.join(current_word)
            tokenized_words.append(word)
        if not tokenized_words:
            raise EncodingError("Tokenization resulted in empty output")
        return tuple(tokenized_words)


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
        return f'{self.__class__.__name__}(name={self.get_name()}, value={self.get_value()})'

    def add_child(self, item: int) -> None:
        """
        Add a new child node with the given item.

        Args:
            item (int): Data value for the new child node.
        """
        self._children.append(TrieNode(name=item))

    def get_children(self, item: int | None = None) -> tuple["TrieNode", ...]:
        """
        Get the tuple of child nodes or one child.

        Args:
            item (int | None, optional): Special data to find special child

        Returns:
            tuple["TrieNode", ...]: Tuple of child nodes.
        """
        if isinstance(item, int):
            return tuple(child for child in self._children if child.get_name() == item)
        return tuple(self._children)

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
        return self.__bool__()


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
        for token in encoded_corpus:
            self._insert(token)

    def get_prefix(self, prefix: NGramType) -> TrieNode:
        """
        Find the node corresponding to a prefix.

        Args:
            prefix (NGramType): Prefix to find trie by.

        Returns:
            TrieNode: Found TrieNode by prefix
        """
        current_node = self._root
        for element in prefix:
            found_element = None
            for child in current_node.get_children():
                if child.get_name() == element:
                    found_element = child
                    break
            if found_element is None:
                raise TriePrefixNotFoundError('prefix is not in the tree')
            current_node = found_element
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
        stack = [(prefix_node, list(prefix))]
        while stack:
            current_node, current_sequence = stack.pop()
            if current_node.has_children():
                children = list(current_node.get_children())
                for child in reversed(children):
                    if child.get_name() is None:
                        continue
                    new_sequence = current_sequence + [child.get_name()]
                    if new_sequence:
                        sequences.append(tuple(new_sequence))
                        stack.append((child, new_sequence))
        return tuple(sequences[::-1])

    def _insert(self, sequence: NGramType) -> None:
        """
        Inserts a token in PrefixTrie

        Args:
            sequence (NGramType): Tokens to insert.
        """
        current_node = self._root
        for token in sequence:
            found_child = False
            for child in current_node.get_children():
                if child.get_name() == token:
                    current_node = child
                    found_child = True
                    break
            if not found_child:
                new_node = TrieNode(token)
                current_node._children.append(new_node)
                current_node = new_node


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
        self._root = TrieNode(value=0.0)

    def __str__(self) -> str:
        """
        Return a string representation of the NGramTrieLanguageModel.

        Returns:
            str: String representation showing n-gram size.
        """
        return f"{self.__class__.__name__}({self._n_gram_size})"

    def build(self) -> int:
        """
        Build the trie using sliding n-gram windows from a tokenized corpus.

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1
        """
        self._root = TrieNode(value=0.0)
        if self._encoded_corpus is None:
            return 1
        try:
            all_ngrams = []
            for sentence in self._encoded_corpus:
                if len(sentence) >= self._n_gram_size:
                    for i in range(len(sentence) - self._n_gram_size + 1):
                        ngram = tuple(sentence[i:i + self._n_gram_size])
                        all_ngrams.append(ngram)
            for ngram in all_ngrams:
                self._insert(ngram)
            collected_ngrams = self._collect_all_ngrams()
            self._fill_frequencies(collected_ngrams)
            return 0
        except Exception:
            return 1

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
        context = tuple(sequence[-(self._n_gram_size - 1):])
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
        if self._encoded_corpus is None or not self._encoded_corpus:
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
            for child in current_node.get_children():
                child_name = child.get_name()
                if child_name is not None:
                    stack.append((child, current_path + [child_name]))
        return tuple(ngrams)


    def _collect_frequencies(self, node: TrieNode) -> dict[int, float]:
        """
        Collect frequencies from immediate child nodes only.

        Args:
            node (TrieNode): Current node.

        Returns:
            dict[int, float]: Collected frequencies of items.
        """
        children = node.get_children()
        frequencies = {}
        for child in children:
            name = child.get_name()
            value = child.get_value()
            if name is not None:
                frequencies[name] = value
        return frequencies

    def _fill_frequencies(self, encoded_corpus: tuple[NGramType, ...]) -> None:
        """
        Calculate and assign frequencies for nodes in the trie based on corpus statistics.

        Counts occurrences of each n-gram and stores the relative frequency on the last node
        of each n-gram sequence.

        Args:
            encoded_corpus (tuple[NGramType, ...]): Tuple of n-grams extracted from the corpus.
        """
        ngram_counts = {}
        for ngram in encoded_corpus:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        total_count = len(encoded_corpus)
        for ngram, ngram_count in ngram_counts.items():
            try:
                node = self.get_prefix(ngram)
                relative_frequency = ngram_count / total_count
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

    def build(self) -> int:
        """
        Build N-gram tries for all possible ngrams based on a corpus of tokens.

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1.
        """

    def set_current_ngram_size(self, current_n_gram_size: int | None) -> None:
        """
        Set the active N-gram size used for generation.

        Args:
            current_n_gram_size (int | None): Current N-gram size for generation.
        """

    def generate_next_token(self, sequence: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve tokens that can continue the given sequence along with their probabilities.

        Args:
            sequence (tuple[int, ...]): A sequence to match beginning of N-grams for continuation.

        Returns:
            dict[int, float] | None: Possible next tokens with their probabilities.
        """

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

    def _merge(self) -> None:
        """
        Merge all built N-gram trie models into a single unified trie.
        """

    def _insert_trie(self, source_root: TrieNode) -> None:
        """
        Insert all nodes of source root trie into our main root.

        Args:
            source_root (TrieNode): Source root to insert tree
        """


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

    def get_next_token(self, sequence_to_continue: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            dict[int, float] | None: Next tokens for sequence continuation
        """

    def run(self, seq_len: int, prompt: str) -> str | None:
        """
        Generate sequence based on dynamic N-gram trie and prompt provided.

        Args:
            seq_len (int): Number of tokens to generate
            prompt (str): Beginning of sequence

        Returns:
            str | None: Generated sequence
        """


def save(trie: DynamicNgramLMTrie, path: str) -> None:
    """
    Save DynamicNgramLMTrie.

    Args:
        trie (DynamicNgramLMTrie): Trie for saving
        path (str): Path for saving
    """


def load(path: str) -> DynamicNgramLMTrie:
    """
    Load DynamicNgramLMTrie from file.

    Args:
        path (str): Trie path

    Returns:
        DynamicNgramLMTrie: Trie from file.
    """
