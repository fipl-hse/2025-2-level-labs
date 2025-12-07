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
            raise EncodingError('Invalid input: text must be a non-empty string')
        eos = self._end_of_sentence_token
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
                cleaned_word = ''.join(x for x in word.lower() if x.isalpha() or x == "'")
                if cleaned_word:
                    tokens.append(cleaned_word)
            tokens.append(eos)
        text_ends_with_punctuation = text.strip() and text.strip()[-1] in '.!?'
        if not text_ends_with_punctuation and tokens and tokens[-1] == eos:
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

    def __init__(self,name: int | None = None, value: float = 0.0, children: list["TrieNode"] | None = None,
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
        sequences = []
        prefixes = list(prefix)
        stack = [(prefix_node, prefixes)]
        while stack:
            node, path = stack.pop()
            children = list(node.get_children())
            results = []
            for child in children:
                child_name = child.get_name()
                if child_name is None:
                    continue
                new_path = path + [child_name]
                if not child.has_children():
                    results.append(tuple(new_path))
                else:
                    stack.append((child, new_path))
            sequences.extend(sorted(results))
        return tuple(sequences)

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
        ngrams = len(encoded_corpus)
        counts = {}
        for ngram in encoded_corpus:
            counts[ngram] = counts.get(ngram, 0) + 1
        for ngram, count in counts.items():
            frequence = count/ngrams
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
        if not isinstance(self._max_ngram_size, int):
            return 1
        if not isinstance(self._encoded_corpus, tuple):
            return 1
        if not self._encoded_corpus:
            return 1
        for size in range(2, self._max_ngram_size + 1):
            model = NGramTrieLanguageModel(self._encoded_corpus, size)
            result = model.build()
            if result == 0:
                self._models[size] = model
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
        size = min(self._current_n_gram_size - 1, len(sequence))
        if size <= 0:
            return {}
        context = sequence[-size:]
        children = False
        try:
            current_node = self._root
            for item in context:
                for child in current_node.get_children():
                    if child.get_name() == item:
                        children = True
                        current_node = child
                        break
                if not children:
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
        if (
            node_name is None or not isinstance(node_name, int) or node_name < 0):
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
        if not source_root.has_children():
            return
        stack = [(source_root, self._root)]
        while stack:
            source_node, node = stack.pop()
            children = source_node.get_children()
            for source_child in children:
                child_name = source_child.get_name()
                if child_name is not None:
                    target_child = self._assign_child(
                        node,
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
        return ''.join(words)

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
        trie_data = json.load(file)
    encoded_corpus = tuple(trie_data.get('encoded_corpus', ()))
    max_ngram_size = trie_data.get('max_ngram_size', 3)
    loaded_trie = DynamicNgramLMTrie(encoded_corpus, max_ngram_size)
    result = loaded_trie.build()
    if result != 0:
        return DynamicNgramLMTrie(tuple(), max_ngram_size)
    loaded_trie.set_current_ngram_size(trie_data.get('current_n_gram_size', max_ngram_size))
    return loaded_trie
