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
    Exception raised when a requested prefix is not found in the trie.
    
    This error occurs when attempting to access a node corresponding to a specific prefix
    that does not exist in the prefix tree structure.
    """


class EncodingError(Exception):
    """
    Exception raised when text encoding fails.
    
    This error occurs during text processing when the input text is invalid,
    empty, or of incorrect type, preventing successful encoding into token sequences.
    """


class DecodingError(Exception):
    """
    Exception raised when token decoding fails.
    
    This error occurs during text reconstruction when the decoded corpus is invalid,
    empty, or results in malformed output during postprocessing.
    """


class IncorrectNgramError(Exception):
    """
    Exception raised for invalid n-gram parameters.
    
    This error occurs when an n-gram size is specified that is less than 1,
    which violates the fundamental requirements for n-gram language modeling.
    """


class MergeTreesError(Exception):
    """
    Exception raised when trie merging operations fail.
    
    This error occurs during attempts to merge multiple prefix trees when
    the operation cannot be completed due to structural inconsistencies
    or incompatible tree configurations.
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
        TextProcessor.__init__(self, end_of_sentence_token)
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
        if not isinstance(text, str) or not text:
            return tuple()
        sentences = []
        sentence_chars = []
        for char in text:
            sentence_chars.append(char)
            if char in '.!?':
                sentence_text = ''.join(sentence_chars).strip()
                sentence_chars = []
                if sentence_text:
                    sentences.append(sentence_text.strip().lower())
        if sentence_chars:
            sentence_text = ''.join(sentence_chars).strip()
            if sentence_text:
                sentences.append(sentence_text.strip().lower())
        encoded_sentences = []
        for sentence in sentences:
            if not sentence:
                continue
            tokens = self._tokenize(sentence)
            encoded_sentence = []
            for token in tokens:
                if token == self._end_of_sentence_token:
                    continue
                self._put(token)
                word_id = self._storage.get(token)
                if word_id is not None:
                    encoded_sentence.append(word_id)
            if encoded_sentence:
                encoded_sentences.append(tuple(encoded_sentence) + (0,))
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
            raise DecodingError('Invalid input: decoded_corpus must be a non-empty tuple')
        if all(token == self._end_of_sentence_token for token in decoded_corpus):
            raise DecodingError('Postprocessing resulted in empty output')
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
            raise DecodingError('Postprocessing resulted in empty output')
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
            raise EncodingError('Invalid input: text must be a non-empty string')
        eos_token = self._end_of_sentence_token
        sentences = []
        current_sentence = []
        index = 0
        text_length = len(text)
        while index < text_length:
            char = text[index]
            if char in '.!?':
                if current_sentence:
                    sentences.append(''.join(current_sentence).strip())
                    current_sentence = []
                while index < text_length and text[index] in '.!? ':
                    index += 1
                continue
            current_sentence.append(char)
            index += 1
        if current_sentence:
            sentences.append(''.join(current_sentence).strip())
        tokens = []
        for sentence in sentences:
            if not sentence:
                continue
            words = sentence.split()
            for word in words:
                cleaned_word = ''.join(c for c in word.lower() if c.isalpha() or c == "'")
                if cleaned_word:
                    tokens.append(cleaned_word)
            tokens.append(eos_token)
        text_ends_with_punctuation = text.strip() and text.strip()[-1] in '.!?'
        if not text_ends_with_punctuation and tokens and tokens[-1] == eos_token:
            tokens = tokens[:-1]
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
        if item is not None:
            new_child = TrieNode(name=item, value=0.0)
            self._children.append(new_child)

    def get_children(self, item: int | None = None) -> tuple["TrieNode", ...]:
        """
        Get the tuple of child nodes or one child.

        Args:
            item (int | None, optional): Special data to find special child

        Returns:
            tuple["TrieNode", ...]: Tuple of child nodes.
        """
        children = tuple(self._children)
        if item is None:
            return children
        return tuple(child for child in children if child.get_name() == item)

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
        if not prefix or not isinstance(prefix, tuple):
            raise ValueError("Invalid prefix")
        current_node = self._root
        for item in prefix:
            children = current_node.get_children(item)
            if not children:
                raise TriePrefixNotFoundError(f'Prefix {prefix} not found in trie')
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
            prefix_node = self.get_prefix(prefix)
        except TriePrefixNotFoundError:
            return tuple()
        sequences = []
        stack: list[tuple[TrieNode, list[int]]] = [(prefix_node, list(prefix))]
        while stack:
            current_node, current_sequence = stack.pop()
            if current_node.has_children():
                children = list(current_node.get_children())
                for child in children[::-1]:
                    child_name = child.get_name()
                    if child_name is None:
                        continue
                    new_sequence = current_sequence + [child_name]
                    sequences.append(tuple(new_sequence))
                    stack.append((child, new_sequence))
        return tuple(sequences[::-1])

    def _insert(self, sequence: NGramType) -> None:
        """
        Inserts a token in PrefixTrie

        Args:
            sequence (NGramType): Tokens to insert.
        """
        current = self._root
        for token in sequence:
            children = current.get_children(token)
            if not children:
                current.add_child(token)
                children = current.get_children(token)
            current = children[0]


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
        return f'NGramTrieLanguageModel({self._n_gram_size})'

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
            for i in range(len(sentence) - self._n_gram_size + 1):
                ngram = tuple(sentence[i:i + self._n_gram_size])
                all_ngrams.append(ngram)
        try:
            for ngram in all_ngrams:
                self._insert(ngram)
            final_ngrams = self._collect_all_ngrams()
            self._fill_frequencies(final_ngrams)
            return 0
        except (ValueError, TriePrefixNotFoundError, KeyError, TypeError):
            return 1

    def get_next_tokens(self, start_sequence: NGramType) -> dict[int, float]:
        """
        Get all possible next tokens and their relative frequencies for a given prefix.

        Args:
            start_sequence (NGramType): The prefix sequence.

        Returns:
            dict[int, float]: Mapping of token → relative frequency.
        """
        prefix_node = self.get_prefix(start_sequence)
        if not prefix_node.has_children():
            return {}
        return self._collect_frequencies(prefix_node)

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
        if (
            not isinstance(sequence, tuple)
            or not sequence
            or len(sequence) < (self._n_gram_size - 1)
        ):
            return None
        context = sequence[-(self._n_gram_size - 1):]
        try:
            generating_next_token = self.get_next_tokens(context)
        except TriePrefixNotFoundError:
            return {}
        return generating_next_token

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
        if not isinstance(new_corpus, tuple) or not new_corpus:
            return
        if not self._encoded_corpus:
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
        all_ngrams = []
        stack = [(self._root, [])]
        while stack:
            node, current_path = stack.pop()
            node_name = node.get_name()
            if node_name is not None:
                new_path: list[int] = current_path + [node_name]
            else:
                new_path = current_path
            if len(new_path) == self._n_gram_size:
                all_ngrams.append(tuple(new_path))
                continue
            for child in node.get_children():
                stack.append((child, new_path))
        return tuple(all_ngrams)

    def _collect_frequencies(self, node: TrieNode) -> dict[int, float]:
        """
        Collect frequencies from immediate child nodes only.

        Args:
            node (TrieNode): Current node.

        Returns:
            dict[int, float]: Collected frequencies of items.
        """
        frequencies = {}
        for child_node in node.get_children():
            token = child_node.get_name()
            if not token:
                continue
            frequency = child_node.get_value()
            frequencies[token] = frequency
        return frequencies

    def _fill_frequencies(self, encoded_corpus: tuple[NGramType, ...]) -> None:
        """
        Calculate and assign frequencies for nodes in the trie based on corpus statistics.

        Counts occurrences of each n-gram and stores the relative frequency on the last node
        of each n-gram sequence.

        Args:
            encoded_corpus (tuple[NGramType, ...]): Tuple of n-grams extracted from the corpus.
        """
        total_ngrams = len(encoded_corpus)
        ngram_count = {}
        for ngram in encoded_corpus:
            ngram_count[ngram] = ngram_count.get(ngram, 0) + 1
        for ngram, count in ngram_count.items():
            relative_freq = count / total_ngrams
            try:
                node = self.get_prefix(ngram)
                node.set_value(relative_freq)
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
        self._models = {}
        self._max_ngram_size = n_gram_size

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
            result = model.build()
            if result != 0:
                return 1
            self._models[ngram_size] = model
        self._merge()
        return 0

    def set_current_ngram_size(self, current_n_gram_size: int | None) -> None:
        """
        Set the active N-gram size used for generation.

        Args:
            current_n_gram_size (int | None): Current N-gram size for generation.
        """
        if (
            not check_positive_int(current_n_gram_size)
            or not current_n_gram_size
            or current_n_gram_size < 2
            or self._max_ngram_size < current_n_gram_size
        ):
            raise IncorrectNgramError('Invalid n-gram size')
        self._current_n_gram_size = current_n_gram_size

    def generate_next_token(self, sequence: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve tokens that can continue the given sequence along with their probabilities.

        Args:
            sequence (tuple[int, ...]): A sequence to match beginning of N-grams for continuation.

        Returns:
            dict[int, float] | None: Possible next tokens with their probabilities.
        """
        if not (isinstance(sequence, tuple) and sequence):
            return None
        if not 2 <= self._current_n_gram_size <= self._max_ngram_size:
            self._current_n_gram_size = self._max_ngram_size
        if self._current_n_gram_size not in self._models:
            for n in range(self._max_ngram_size, 1, -1):
                if n in self._models:
                    self._current_n_gram_size = n
                    break
            else:
                return {}
        model = self._models[self._current_n_gram_size]
        ngram_size = model.get_n_gram_size()
        if len(sequence) < ngram_size - 1:
            return {}
        context_size = min(self._current_n_gram_size - 1, len(sequence))
        context = sequence[-context_size:]
        try:
            prefix_node = self.get_prefix(context)
            result = {}
            for child in prefix_node.get_children():
                child_name = child.get_name()
                if child_name is not None:
                    result[child_name] = child.get_value()
            return result
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
        if (
            node_name is None
            or not isinstance(node_name, int)
            or node_name < 0
        ):
            raise ValueError('Node name must be non-negative integer')
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
            raise MergeTreesError('No models to merge')
        self._root = TrieNode()
        for n_size in sorted(self._models):
            model = self._models[n_size]
            source_root = model.get_root()
            self._insert_trie(source_root)

    def _insert_trie(self, source_root: TrieNode) -> None:
        """
        Insert all nodes of source root trie into our main root.

        Args:
            source_root (TrieNode): Source root to insert tree
        """
        if not source_root.has_children():
            return
        stack = [(source_root, self._root)]
        while stack:
            source_node, target_node = stack.pop()
            children = source_node.get_children()
            for source_child in children:
                child_name = source_child.get_name()
                if child_name is not None:
                    target_child = self._assign_child(
                        target_node,
                        child_name,
                        source_child.get_value()
                    )
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
        BackOffGenerator.__init__(self, (dynamic_trie,), processor)
        self._dynamic_trie = dynamic_trie

    def get_next_token(self, sequence_to_continue: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            dict[int, float] | None: Next tokens for sequence continuation
        """
        if (
            not isinstance(sequence_to_continue, tuple)
            or len(sequence_to_continue) == 0
        ):
            return None
        max_size = self._dynamic_trie.get_n_gram_size()
        ngram_sizes = list(range(max_size, 1, -1))
        for ngram in ngram_sizes:
            self._dynamic_trie.set_current_ngram_size(ngram)
            next_tokens = self._dynamic_trie.generate_next_token(sequence_to_continue)
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
        if (
            not check_positive_int(seq_len)
            or not isinstance(prompt, str)
            or len(prompt) == 0
        ):
            return None
        try:
            encoded_seq = self._text_processor.encode(prompt)
        except EncodingError:
            return None
        if not encoded_seq:
            return None
        tokens = list(encoded_seq)
        eos_token_id = getattr(self._text_processor, '_end_of_word_token', None)
        if eos_token_id is None:
            eos_token_id = self._text_processor.get_end_of_word_token()
        if tokens and tokens[-1] == eos_token_id:
            tokens.pop()
        for _ in range(seq_len):
            next_tokens = self.get_next_token(tuple(tokens))
            if not next_tokens:
                break
            best_token = max(next_tokens.items(), key=lambda x: (x[1], x[0]))[0]
            tokens.append(best_token)
        words = []
        storage = getattr(self._text_processor, '_storage', None)
        for token_id in tokens:
            for word, word_id in storage.items():
                if word_id == token_id:
                    words.append(word)
                    break
        postprocess_method = getattr(self._text_processor, '_postprocess_decoded_text', None)
        if postprocess_method:
            return str(postprocess_method(tuple(words)))
        return ' '.join(words)

def save(trie: DynamicNgramLMTrie, path: str) -> None:
    """
    Save DynamicNgramLMTrie.

    Args:
        trie (DynamicNgramLMTrie): Trie for saving
        path (str): Path for saving
    """
    if not isinstance(path, str) or not path:
        raise ValueError('Invalid path')
    data = {
    'value': trie.get_root().get_name(),
    'freq': trie.get_root().get_value(),
    'children': []
    }
    stack = [(trie.get_root(), data['children'])]
    while stack:
        current_node, parent_children_list = stack.pop()
        children = current_node.get_children()
        for child in children:
            child_dict = {
                'value': child.get_name(),
                'freq': child.get_value(),
                'children': []
            }
            parent_children_list.append(child_dict)
            stack.append((child, child_dict['children']))
    trie_data = {'trie': data}
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(trie_data, file, indent=2)

def load(path: str) -> DynamicNgramLMTrie:
    """
    Load DynamicNgramLMTrie from file.

    Args:
        path (str): Trie path

    Returns:
        DynamicNgramLMTrie: Trie from file.
    """
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    encoded_corpus = tuple(data.get('encoded_corpus', ()))
    max_ngram_size = data.get('max_ngram_size', 3)
    loaded_trie = DynamicNgramLMTrie(encoded_corpus, max_ngram_size)
    result = loaded_trie.build()
    if result != 0:
        return DynamicNgramLMTrie(tuple(), max_ngram_size)
    loaded_trie.set_current_ngram_size(data.get('current_n_gram_size', max_ngram_size))
    return loaded_trie