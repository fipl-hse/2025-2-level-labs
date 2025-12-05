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
    Exception raised when a requested prefix is not found in the trie.
    
    This error occurs when attempting to access a node corresponding to a specific prefix
    that does not exist in the prefix tree structure.
    """
    pass


class EncodingError(Exception):
    """
    Exception raised when text encoding fails.
    
    This error occurs during text processing when the input text is invalid,
    empty, or of incorrect type, preventing successful encoding into token sequences.
    """
    pass


class DecodingError(Exception):
    """
    Exception raised when token decoding fails.
    
    This error occurs during text reconstruction when the decoded corpus is invalid,
    empty, or results in malformed output during postprocessing.
    """
    pass


class IncorrectNgramError(Exception):
    """
    Exception raised for invalid n-gram parameters.
    
    This error occurs when an n-gram size is specified that is less than 1,
    which violates the fundamental requirements for n-gram language modeling.
    """
    pass


class MergeTreesError(Exception):
    """
    Exception raised when trie merging operations fail.
    
    This error occurs during attempts to merge multiple prefix trees when
    the operation cannot be completed due to structural inconsistencies
    or incompatible tree configurations.
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
        super().__init__(end_of_word_token=end_of_sentence_token)
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
        if not text or not isinstance(text, str):
            raise EncodingError("Invalid input text")
        sentences = []
        current_sentence = []
        tokens = self._tokenize(text)
        for token in tokens:
            if token != self._end_of_sentence_token:
                self._put(token)
                word_id = self._storage[token]
                current_sentence.append(word_id)
                continue
            if current_sentence:
                self._put(self._end_of_sentence_token)
                eos_id = self._storage[self._end_of_sentence_token]
                current_sentence.append(eos_id)
                sentences.append(tuple(current_sentence))
                current_sentence = []
        if current_sentence:
            self._put(self._end_of_sentence_token)
            eos_id = self._storage[self._end_of_sentence_token]
            current_sentence.append(eos_id)
            sentences.append(tuple(current_sentence))
        return tuple(sentences)

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): An element to put into storage

        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """
        if not element or not isinstance(element, str):
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
        if not decoded_corpus or not isinstance(decoded_corpus, tuple):
            raise DecodingError("Invalid input: decoded_corpus must be a non-empty tuple")
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
            if sentence:
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
        if not text or not isinstance(text, str):
            raise EncodingError("Invalid input: text must be a non-empty string")
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
        for child in self._children:
            if child.get_name() == item:
                return False
        self._children.append(TrieNode(name=item))
        return True

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
                raise TriePrefixNotFoundError(f"Prefix {prefix} not found in trie")
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
            return ()
        sequences = []
        stack = [(prefix_node, list(prefix))]
        while stack:
            current_node, current_sequence = stack.pop()
            children = current_node.get_children()
            valid_children_found = False
            for child in children:
                child_name = None
                if hasattr(child, 'get_name'):
                    child_name = child.get_name()
                elif hasattr(child, 'get_data'):
                    child_name = child.get_data()
                if child_name is not None:
                    valid_children_found = True
                    stack.append((child, current_sequence + [child_name]))
            if not valid_children_found and len(current_sequence) > len(prefix):
                sequences.append(tuple(current_sequence))
        if sequences:
            try:
                sequences.sort()
            except TypeError:
                pass
        return tuple(sequences)

    def _insert(self, sequence: NGramType) -> None:
        """
        Inserts a token in PrefixTrie

        Args:
            sequence (NGramType): Tokens to insert.
        """
        if not sequence or not isinstance(sequence, tuple):
            return
        current_node = self._root
        for item in sequence:
            children = current_node.get_children(item)
            if children:
                current_node = children[0]
            else:
                current_node.add_child(item)
                new_children = current_node.get_children(item)
                if new_children:
                    current_node = new_children[0]


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
        PrefixTrie.__init__(self)
        NGramLanguageModel.__init__(self, encoded_corpus, n_gram_size)
        self._n_gram_size = n_gram_size
        self._encoded_corpus = encoded_corpus

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
        self.clean()
        all_ngrams = []
        for sentence in self._encoded_corpus:
            if len(sentence) < self._n_gram_size:
                continue
            for i in range(len(sentence) - self._n_gram_size + 1):
                ngram = sentence[i:i + self._n_gram_size]
                self._insert(ngram)
                all_ngrams.append(ngram)
        if not all_ngrams:
            return 1
        self._fill_frequencies(tuple(all_ngrams))
        return 0

    def get_next_tokens(self, start_sequence: NGramType) -> dict[int, float]:
        """
        Get all possible next tokens and their relative frequencies for a given prefix.

        Args:
            start_sequence (NGramType): The prefix sequence.

        Returns:
            dict[int, float]: Mapping of token → relative frequency.
        """
        prefix_node = self.get_prefix(start_sequence)
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
        if not sequence or not isinstance(sequence, tuple):
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
            self._encoded_corpus = self._encoded_corpus + new_corpus
        self.build()

    def _collect_all_ngrams(self) -> tuple[NGramType, ...]:
        """
        Collect all n-grams from the trie by traversing all paths of length n_gram_size.

        Returns:
            tuple[NGramType, ...]: Tuple of all n-grams stored in the trie.
        """
        sequences = []
        stack = [(self._root, [])]
        while stack:
            current_node, current_path = stack.pop()
            if len(current_path) == self._n_gram_size:
                sequences.append(tuple(current_path))
                continue
            for child in current_node.get_children():
                child_name = child.get_name()
                if child_name is None:
                    continue
                new_path = current_path + [child_name]
                stack.append((child, new_path))
        return tuple(sequences)

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
            return
        ngram_counts = {}
        total_ngrams = len(encoded_corpus)
        for ngram in encoded_corpus:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        for ngram, count in ngram_counts.items():
            relative_frequency = count / total_ngrams
            node = self.get_prefix(ngram)
            node.set_value(relative_frequency)


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
        self._max_ngram_size = n_gram_size
        self._current_n_gram_size = 0
        self._models = {}
        self._encoded_corpus = encoded_corpus

    def build(self) -> int:
        """
        Build N-gram tries for all possible ngrams based on a corpus of tokens.

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1.
        """
        if (
            not isinstance(self._encoded_corpus, tuple)
            or not all(isinstance(sentence, tuple) for sentence in self._encoded_corpus)
            or not isinstance(self._max_ngram_size, int)
            or self._max_ngram_size < 2
        ):
            return 1
        self._models = {}
        for n_size in range(2, self._max_ngram_size + 1):
            model = NGramTrieLanguageModel(self._encoded_corpus, n_size)
            if model.build() == 0:
                self._models[n_size] = model
        if not self._models:
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
        if not isinstance(current_n_gram_size, int):
            raise IncorrectNgramError("Invalid n-gram size")
        if current_n_gram_size < 2 or current_n_gram_size > self._max_ngram_size:
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
        if not isinstance(sequence, tuple) or len(sequence) == 0:
            return None
        if self._current_n_gram_size < 2:
            self._current_n_gram_size = self._max_ngram_size
        if self._current_n_gram_size in self._models:
            model = self._models[self._current_n_gram_size]
            if len(sequence) < model.get_n_gram_size() - 1:
                return {}
            context = sequence[-(model.get_n_gram_size() - 1):]
            try:
                return model.get_next_tokens(context)
            except TriePrefixNotFoundError:
                return {}
        context_size = min(self._current_n_gram_size - 1, len(sequence))
        if context_size <= 0:
            return {}
        context = sequence[-context_size:]
        try:
            prefix_node = self.get_prefix(context)
            frequencies = {}
            for child in prefix_node.get_children():
                child_name = child.get_name()
                if child_name is not None:
                    frequencies[child_name] = child.get_value()
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
        if node_name is None or not isinstance(node_name, int) or node_name < 0:
            raise ValueError("Node name must be a non-negative integer")
        for child in parent._children:
            if child.get_name() == node_name:
                child.set_value(freq)
                return child
        new_child = TrieNode(name=node_name, value=freq)
        parent._children.append(new_child)
        return new_child

    def _merge(self) -> None:
        """
        Merge all built N-gram trie models into a single unified trie.
        """
        if not self._models:
            raise MergeTreesError("No models to merge")
        self._root = TrieNode()
        for n_size in sorted(self._models.keys()):
            self._insert_trie(self._models[n_size].get_root())

    def _insert_trie(self, source_root: TrieNode) -> None:
        """
        Insert all nodes of source root trie into our main root.

        Args:
            source_root (TrieNode): Source root to insert tree
        """
        stack = [(source_root, self._root)]
        while stack:
            source_node, target_node = stack.pop()
            for source_child in source_node.get_children():
                child_name = source_child.get_name()
                if child_name is not None:
                    target_child = self._assign_child(target_node, child_name, source_child.get_value())
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
        if not isinstance(sequence_to_continue, tuple) or len(sequence_to_continue) == 0:
            return None
        max_ngram_size = self._dynamic_trie._max_ngram_size
        sequence_length = len(sequence_to_continue)
        ngram_sizes = list(range(2, max_ngram_size + 1))
        ngram_sizes.sort(reverse=True)
        for n_size in ngram_sizes:
            context_size = min(n_size - 1, sequence_length)
            if context_size <= 0:
                continue
            self._dynamic_trie.set_current_ngram_size(n_size)
            context = sequence_to_continue[-context_size:]
            next_tokens = self._dynamic_trie.generate_next_token(context)
            if next_tokens:
                return next_tokens
        for n_size in ngram_sizes:
            self._dynamic_trie.set_current_ngram_size(n_size)
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
        if not isinstance(seq_len, int) or seq_len <= 0:
            return None
        if not isinstance(prompt, str) or not prompt:
            return None
        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None
        generated_sequence = list(encoded_prompt)
        for _ in range(seq_len):
            next_tokens = self.get_next_token(tuple(generated_sequence))
            if not next_tokens:
                break
            best_token = max(next_tokens.items(), key=lambda x: x[1])[0]
            generated_sequence.append(best_token)
        decoded_words = []
        reverse_storage = {v: k for k, v in self._text_processor._storage.items()}
        for token_id in generated_sequence:
            if token_id in reverse_storage:
                decoded_words.append(reverse_storage[token_id])
        if decoded_words and decoded_words[-1] != self._text_processor._end_of_sentence_token:
            decoded_words.append(self._text_processor._end_of_sentence_token)
        result = self._text_processor._postprocess_decoded_text(tuple(decoded_words))
        return result


def save(trie: DynamicNgramLMTrie, path: str) -> None:
    """
    Save DynamicNgramLMTrie.

    Args:
        trie (DynamicNgramLMTrie): Trie for saving
        path (str): Path for saving
    """
    root_dict = {
        "value": trie._root.get_name(),
        "freq": trie._root.get_value(),
        "children": []
    }
    stack = [(trie._root, root_dict["children"])]
    while stack:
        current_node, children_list = stack.pop()
        for child in current_node.get_children():
            child_dict = {
                "value": child.get_name(),
                "freq": child.get_value(),
                "children": []
            }
            children_list.append(child_dict)
            stack.append((child, child_dict["children"]))
    trie_data = {"trie": root_dict}
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
    empty_trie = DynamicNgramLMTrie((), 3)
    empty_trie._current_n_gram_size = 0
    root_dict = trie_data["trie"]
    empty_trie._root = TrieNode(
        name=root_dict["value"],
        value=root_dict["freq"]
    )
    stack = [(empty_trie._root, root_dict["children"])]
    while stack:
        current_node, children_data = stack.pop()
        for child_data in children_data:
            child_node = TrieNode(
                name=child_data["value"],
                value=child_data["freq"]
            )
            current_node._children.append(child_node)
            stack.append((child_node, child_data["children"]))
    return empty_trie
