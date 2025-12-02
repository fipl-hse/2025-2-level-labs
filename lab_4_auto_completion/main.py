"""
Lab 4
"""

# pylint: disable=unused-argument, super-init-not-called, unused-private-member, duplicate-code, unused-import
import json
import string

from lab_3_generate_by_ngrams.main import BackOffGenerator, NGramLanguageModel, TextProcessor

NGramType = tuple[int, ...]
"Type alias for NGram."

class TriePrefixNotFoundError(Exception):
    """Prefix not found in tree"""
    pass

class EncodingError(Exception):
    """Encoding not passed correctly"""
    pass

class DecodingError(Exception):
    """Input not decoded"""
    pass

class IncorrectNgramError(Exception):
    """Incorrect ngram input"""
    pass

class MergeTreesError(Exception):
    """Trees not merged"""
    pass

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
        self._storage = {end_of_sentence_token: 0}

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
        tokens = self._tokenize(text)
        sentences = []
        encoded_sentences = []
        new_sentence = []

        for token in tokens:
            if token == self._end_of_sentence_token:
                if new_sentence:
                    new_sentence.append(self._end_of_sentence_token)
                    sentences.append(new_sentence)
                    new_sentence = []
            else:
                new_sentence.append(token)

        for sentence in sentences:
            id_sentence = []
            for word in sentence:
                self._put(word)
                id_sentence.append(self._storage[word])
            encoded_sentences.append(tuple(id_sentence))

        return tuple(encoded_sentences)

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): An element to put into storage

        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """
        if isinstance(element, str) and element not in self._storage and element:
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
        check_corpus = False
        for token in decoded_corpus:
            if token != self._end_of_sentence_token or token.isalpha():
                check_corpus = True
                break
        if not check_corpus:
            raise DecodingError("Postprocessing resulted in empty output")

        tokens = list(decoded_corpus)
        processed_tokens = ""
        prev_token_eos = True
        for token in tokens:
            if prev_token_eos:
                token = token.capitalize()
                prev_token_eos = False
            if token == self._end_of_sentence_token:
                processed_tokens = processed_tokens[:-1]
                token = "."
                prev_token_eos = True
            processed_tokens += f"{token} "
        
        processed_tokens = processed_tokens[:-1]
        processed_tokens += "."
        if not processed_tokens:
            raise DecodingError("Postprocessing resulted in empty output")
        return processed_tokens

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

        words = text.lower().split()
        tokens = []

        for word in words:
            new_word = "".join(symbol for symbol in word if symbol.isalpha() or symbol == "-")
            if not new_word:
                continue
            tokens.append(new_word)
            if word and word[-1] in ".!?":
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
        child = TrieNode(item)
        self._children.append(child)

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
        children = tuple(child for child in self._children if child.get_name() == item)
        return children

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
        return bool(self._children)


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
        for ngram in encoded_corpus:
            self._insert(ngram)
        return None

    def get_prefix(self, prefix: NGramType) -> TrieNode:
        """
        Find the node corresponding to a prefix.

        Args:
            prefix (NGramType): Prefix to find trie by.

        Returns:
            TrieNode: Found TrieNode by prefix
        """
        current_node = self._root
        for el in prefix:
            children = current_node.get_children(el)
            if not children:
                raise TriePrefixNotFoundError("")
            current_node = children[0]
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
            current_node = self.get_prefix(prefix)
        except TriePrefixNotFoundError:
            return tuple()
        results = []

        queue = [(current_node, prefix)]
        while queue:
            current_node, current_sequence = queue.pop()
            if current_node.has_children():
                for child in current_node.get_children():
                    child_name = child.get_name()
                    if child_name is None:
                        continue
                    next_sequence = current_sequence + (child_name,)
                    queue.append((child, next_sequence))
            else:
                if current_sequence != prefix:
                    results.append(current_sequence)
        return tuple(results)

    def _insert(self, sequence: NGramType) -> None:
        """
        Inserts a token in PrefixTrie

        Args:
            sequence (NGramType): Tokens to insert.
        """
        if not isinstance(sequence, tuple) or not sequence:
            raise IncorrectNgramError("hi")

        current_node = self._root
        for token in sequence:
            children = current_node.get_children(token)
            if not children:
                current_node.add_child(token)
                children = current_node.get_children(token)
            current_node = children[0]


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
        PrefixTrie.__init__(self)
        self._encoded_corpus = encoded_corpus
        self._root = TrieNode()
        self._n_gram_size = n_gram_size


    def __str__(self) -> str:
        """
        Return a string representation of the NGramTrieLanguageModel.

        Returns:
            str: String representation showing n-gram size.
        """
        return f"NGramTrieLanguageModel: n_gram_size={self._n_gram_size}"

    def build(self) -> int:
        """
        Build the trie using sliding n-gram windows from a tokenized corpus.

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1
        """
        self.clean()
        ngrams = []
        for sentence in self._encoded_corpus:
            if len(sentence) >= self._n_gram_size:
                for i in range(len(sentence)):
                    sized_ngram = sentence[i:i + self._n_gram_size]
                    ngrams.append(sized_ngram)
        if not ngrams:
            return 1

        for ngram in ngrams:
            self._insert(ngram)
        self._fill_frequencies(tuple(ngrams))
        if not self._root.has_children():
            return 1
        return 0

    def get_next_tokens(self, start_sequence: NGramType) -> dict[int, float]:
        """
        Get all possible next tokens and their relative frequencies for a given prefix.

        Args:
            start_sequence (NGramType): The prefix sequence.

        Returns:
            dict[int, float]: Mapping of token → relative frequency.
        """
        try:
            node = self.get_prefix(start_sequence)
            return self._collect_frequencies(node)
        except TriePrefixNotFoundError as e:
            raise TriePrefixNotFoundError from e

    def get_root(self) -> TrieNode:
        """
        Get the root.
        Returns:
            TrieNode: Found root.
        """

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
        if not sequence or not isinstance(sequence, tuple) or len(sequence) < self._n_gram_size - 1:
            return None
        context = sequence[-self._n_gram_size + 1:]
        try:
            next_tokens = self.get_next_tokens(context)
        except TriePrefixNotFoundError:
            return {}
        return next_tokens

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
        self.get_prefix(prefix)

    def update(self, new_corpus: tuple[NGramType]) -> None:
        """
        Update the trie with additional data and refresh frequency values.

        Args:
            new_corpus (tuple[NGramType]): Additional corpus represented as token sequences.
        """
        if self._encoded_corpus is None:
            self._encoded_corpus = new_corpus
        else:
            encoded_corpus = list(self._encoded_corpus)
            new_corpus = list(new_corpus)
            self._encoded_corpus = tuple(encoded_corpus + new_corpus)
        self.build()

    def _collect_all_ngrams(self) -> tuple[NGramType, ...]:
        """
        Collect all n-grams from the trie by traversing all paths of length n_gram_size.

        Returns:
            tuple[NGramType, ...]: Tuple of all n-grams stored in the trie.
        """
        proper_children = []
        queue = [(self._root, ())]

        while queue:
            current_node, current_seq = queue.pop(0)
            if len(current_seq) == self._n_gram_size:
                proper_children.append(current_seq)
                continue

            for child in current_node.get_children():
                child_name = child.get_name()
                if child_name:
                    new_sequence = current_seq + (child_name,)
                queue.append((child, new_sequence))

        return tuple(proper_children)

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
            child_name = child.get_name()
            if child_name is not None:
                frequencies[child_name] = child.get_value()
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
            return None

        freqs = {}
        for ngram in encoded_corpus:
            freqs[ngram] = freqs.get(ngram, 0) + 1
        total = len(encoded_corpus)

        for ngram, abs_freq in freqs.items():
            try:
                node = self.get_prefix(ngram)
                node.set_value(abs_freq/total)
            except TriePrefixNotFoundError:
                continue
        return None


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
