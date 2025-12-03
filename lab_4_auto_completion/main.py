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
    Exception is raised when the prefix required for the transition is missing from the tree.
    """


class EncodingError(Exception):
    """
    Exception is raised when text encoding fails due to incorrect input or processing error.
    """


class DecodingError(Exception):
    """
    Exception is raised when text decoding fails due to incorrect input or processing error.
    """


class IncorrectNgramError(Exception):
    """
    Exception is raised when trying to use an unsuitable n-gram size.
    """


class MergeTreesError(Exception):
    """
    Exception is raised when it is impossible to merge trees.
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
            return None
        tokens = self._tokenize(text)
        current_sentence = []
        encoded_sentences = []
        for token in tokens:
            if token == self._end_of_sentence_token:
                if current_sentence:
                    encoded_sentences.append(tuple(current_sentence) + (0,))
                    current_sentence = []
            else:
                self._put(token)
                word_id = self._storage.get(token)
                if word_id is not None:
                    current_sentence.append(word_id)
        if current_sentence:
            encoded_sentences.append(tuple(current_sentence) + (0,))
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
            raise DecodingError('Invalid input: decoded_corpus must be a non-empty tuple')
        if all(token == self._end_of_sentence_token for token in decoded_corpus):
            raise DecodingError('Postprocessing resulted in empty output')
        tokens = list(decoded_corpus)
        tokens[0] = tokens[0].capitalize()
        if self.get_end_of_word_token() in tokens:
            for index in range(1, len(tokens)):
                if tokens[index - 1] == self.get_end_of_word_token():
                    tokens[index] = tokens[index].capitalize()
        if not tokens:
            raise DecodingError('Postprocessing resulted in empty output')
        text = ' '.join(tokens).replace(' ' + self.get_end_of_word_token(), '.')
        if text[-1] != '.':
            text += '.'
        return text

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
        words = text.lower().split()
        tokens = []
        for word in words:
            letters = ''.join(letter for letter in word if letter.isalpha() or letter == '-')
            if letters:
                tokens.append(letters)
                if word and word[-1] in '!?.':
                    tokens.append(self._end_of_sentence_token)
        if not tokens:
            raise EncodingError('Tokenization resulted in empty output')
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
        for element in encoded_corpus:
            self._insert(element)


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
            found_child = None
            for child in current_node.get_children():
                if child.get_name() == item:
                    found_child = child
                    break
            if found_child is None:
                raise TriePrefixNotFoundError(f'Prefix {prefix} not found in trie')
            current_node = found_child
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
            node = self.get_prefix(prefix)
        except TriePrefixNotFoundError:
            return []
        results = []
        queue = [(node, prefix)]
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
        current_node = self._root
        for element in sequence:
            children = current_node.get_children()
            child_node = None
            for child in children:
                if child.get_name() == element:
                    child_node = child
                    break
            if child_node is None:
                current_node.add_child(element)
                updated_children = current_node.get_children()
                for child in updated_children:
                    if child.get_name() == element:
                        child_node = child
                        break
            current_node = child_node


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
        return f'NGramTrieLanguageModel({self._n_gram_size})'

    def build(self) -> int:
        """
        Build the trie using sliding n-gram windows from a tokenized corpus.

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1
        """
        attempt = 1
        if attempt:
            print('Start...')
            self._root = TrieNode()
            print(f'The corpus has {len(self._encoded_corpus)} sentences')
            all_ngrams = []
            for sentence in self._encoded_corpus:
                if not isinstance(sentence, (tuple, list)) or not sentence:
                    print(f'Skipping invalid sentence: {sentence} (type: {type(sentence)})')
                    continue
                if len(sentence) < self._n_gram_size:
                    continue
                for i in range(len(sentence) - self._n_gram_size + 1):
                    ngram = tuple(sentence[i:i + self._n_gram_size])
                    all_ngrams.append(ngram)
            print(f'There are {len(all_ngrams)} ngrams')
            self.fill(tuple(all_ngrams))
            print('The tree is done')
            collected_ngrams = self._collect_all_ngrams()
            print(f'Collected {len(collected_ngrams)} ngrams from the tree')
            self._fill_frequencies(collected_ngrams)
            print('Frequencies completed')
            return 0
        else:
            print('Error:')
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
        if (
            not isinstance(sequence, tuple)
            or not sequence
            or len(sequence) < (self._n_gram_size - 1)
        ):
            return None
        context = sequence[:self._n_gram_size - 1]
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
            return None
        if not self._encoded_corpus:
            self._encoded_corpus = new_corpus
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
                new_path = current_path + [node_name]
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
        freq_dict = {}
        for child_node in node.get_children():
            token = child_node.get_name()
            if not token:
                continue
            frequency = child_node.get_value()
            freq_dict[token] = frequency
        return freq_dict

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
        NGramTrieLanguageModel.__init__(self, encoded_corpus, n_gram_size)
        self._root = TrieNode()
        self._encoded_corpus = encoded_corpus
        self._current_n_gram_size = 0
        self._max_ngram_size = n_gram_size
        self._models = {}

    def build(self) -> int:
        """
        Build N-gram tries for all possible ngrams based on a corpus of tokens.

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1.
        """
        if (
            not hasattr(self, '_encoded_corpus')
            or not isinstance(self._encoded_corpus, tuple)
            or not self._encoded_corpus
        ):
            return 1
        for sentence in self._encoded_corpus:
            if not isinstance(sentence, tuple):
                return 1
        if (
            not hasattr(self, '_max_ngram_size')
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
            or  current_n_gram_size < 2
            or self._max_ngram_size < current_n_gram_size
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
        if not check_positive_int(node_name):
            raise ValueError('Oops')
        children = parent.get_children()
        for child in children:
            if child.get_name() == node_name:
                if freq != 0.0:
                    child.set_value(freq)
                return child
        new_child = TrieNode(node_name)
        if freq != 0.0:
            new_child.set_value(freq)
        parent.add_child(new_child)
        return new_child

    def _merge(self) -> None:
        """
        Merge all built N-gram trie models into a single unified trie.
        """
        new_node = TrieNode()
        if not self._models:
            raise MergeTreesError
        if isinstance(self._models, dict):
            sorted_models = [self._models[n] for n in sorted(self._models.keys())]
        for model in sorted_models:
            if hasattr(model, '_root'):
                source_root = model._root
                self._insert_trie(source_root)

    def _insert_trie(self, source_root: TrieNode) -> None:
        """
        Insert all nodes of source root trie into our main root.

        Args:
            source_root (TrieNode): Source root to insert tree
        """
        if not source_root:
            return
        stack = [(source_root, self._root)]
        while stack:
            source_node, destination_node = stack.pop()
            source_name = source_node.get_name()
            if source_name is not None:
                source_value = source_node.get_value()
                try:
                    next_dest_node = self._assign_child(destination_node, source_name, source_value)
                    destination_node = next_dest_node
                except ValueError:
                    continue
            children = source_node.get_children()
            for child in children:
                stack.append((child, destination_node))


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
