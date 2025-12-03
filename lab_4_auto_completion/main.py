"""
Lab 4
"""

# pylint: disable=unused-argument, super-init-not-called, unused-private-member, duplicate-code, unused-import
import json

from lab_3_generate_by_ngrams.main import BackOffGenerator, NGramLanguageModel, TextProcessor

NGramType = tuple[int, ...]
"Type alias for NGram."

class TextProcessingError(Exception):
    """Base exception for errors occurng during text processing."""
    pass

class EncodingError(TextProcessingError):
    """
    Raised when text encoding fails due to invalid input, usupoted
    encoding or processing issues.
    """
    pass

class DecodingError(TextProcessingError):
    """
    Raised when text decoding fails due to invalid input, unsupported
    encoding, or processing issues.
    """
    pass

class TrieError(Exception):
    pass

class TriePrefixNotFoundError(TrieError):
    """
    Raised when the required prefix for transition
    is not found in the trie.
    """
    pass

class MergeTreesError(TrieError):
    """Raised when there is an error merging trees."""
    pass

class IncorrectNgramError(Exception):
    """
    Raised when attempting to use an inappropriate n-gram size.
    """
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
        sentences = []
        sentence = []
        for i in super().encode(text):
            sentence.append(i)
            if i == 0:
                if sentence:
                    sentences.append(tuple(sentence))
                    sentence = []
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
        for i in decoded_corpus:
            if i == self._end_of_sentence_token:
                if sentence:
                    processed_text.append(" ".join(sentence).capitalize())
                    sentence.clear()
            else:
                sentence.append(i)
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
        sentences = []
        sentence = []
        for token in text:
            sentence.append(token)
            if token in "!?.":
                if sentence:
                    sentences.append("".join(sentence).lower())
                    sentence.clear()
        if sentence:
            sentences.append("".join(sentence).lower())
        tokens = []
        for sentence in sentences:
            if isinstance(sentence, str):
                for word in sentence.split():
                    cleaned_word = "".join(symbol for symbol in word if symbol.isalpha())
                    if cleaned_word:
                        tokens.append(cleaned_word)
                if cleaned_word:
                    tokens.append(self._end_of_word_token)
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

    def __init__(self,name: int | None = None,
                 value: float = 0.0,
                 children: list["TrieNode"] | None = None,
        ) -> None:
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
        if not isinstance(item, int):
            raise ValueError
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
        new_node = self._root
        for token in prefix:
            children = new_node.get_children(token)
            if not children:
                raise TriePrefixNotFoundError
            new_node = children[0]
        return new_node


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
        results = []
        queue = [(tuple(prefix), prefix_node)]
        prefix_len = len(prefix)
        while queue:
            new_prefix, new_node = queue.pop(0)
            children = new_node.get_children()
            if not children:
                if len(new_prefix) > prefix_len:
                    results.append(new_prefix)
                continue
            for child in children:
                name = child.get_name()
                if name is not None:
                    queue.append((new_prefix + (name,), child))
                else:
                    queue.append((new_prefix, child))
        return tuple(sorted(results, key=lambda x: x, reverse=True))

    def _insert(self, sequence: NGramType) -> None:
        """
        Inserts a token in PrefixTrie
        Args:
            sequence (NGramType): Tokens to insert.
        """
        new_node = self._root
        for token in sequence:
            children_with_token = new_node.get_children(token)
            if children_with_token:
                new_node = children_with_token[0]
            else:
                new_node.add_child(token)
                new_node = new_node.get_children(token)[0]

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
            self._encoded_corpus + new_corpus

        self.build()

    def _collect_all_ngrams(self) -> tuple[NGramType, ...]:
        """
        Collect all n-grams from the trie by traversing all paths of length n_gram_size.
        Returns:
            tuple[NGramType, ...]: Tuple of all n-grams stored in the trie.
        """
        first_children = self._root.get_children()
        if not first_children:
            return tuple()
        queue = []
        for child in first_children:
            queue.append(((child.get_name(), ), child))
        results = []
        while queue:
            current_ngram, node = queue.pop(0)
            if isinstance(node, TrieNode):
                children = node.get_children()
            else:
                children = None
            if not children:
                continue
            for child in children:
                ngram = current_ngram + (child.get_name(),)
                if len(ngram) == self._n_gram_size:
                    results.append(ngram)
                else:
                    queue.append((ngram, node))
        return tuple(results)

    def _collect_frequencies(self, node: TrieNode) -> dict[int, float]:
        """
        Collect frequencies from immediate child nodes only.
        Args:
            node (TrieNode): Current node.
        Returns:
            dict[int, float]: Collected frequencies of items.
        """
        return {child.get_name(): child.get_value()
                for child in node.get_children()
                if child.get_name() is not None}

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
        ngrams = len(encoded_corpus)
        ngram_counts = {}
        for ngram in encoded_corpus:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        for ngram, count in ngram_counts.items():
            frequence = count / ngrams
            try:
                self.get_prefix(ngram).set_value(frequence)
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
                source_child_name = source_child.get_name()
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
