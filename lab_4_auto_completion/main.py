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
        super().__init__(end_of_sentence_token)
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
        if not isinstance(text, str):
            raise EncodingError("Invalid input: text must be a string")
        if not text.strip():
            raise EncodingError("Invalid input: text must be a non-empty string")
        token_sequence = self._tokenize(text)
        for token in token_sequence:
            if token != self._end_of_sentence_token:
                self._put(token)
        encoded_sentences = []
        current_encoded_sentence = []
        for token in token_sequence:
            if token == self._end_of_sentence_token:
                if current_encoded_sentence:
                    eos_id = self.get_id(self._end_of_sentence_token)
                    current_encoded_sentence.append(eos_id)
                    encoded_sentences.append(tuple(current_encoded_sentence))
                    current_encoded_sentence = []
            else:
                word_id = self.get_id(token)
                current_encoded_sentence.append(word_id)
        if current_encoded_sentence:
            eos_id = self.get_id(self._end_of_sentence_token)
            current_encoded_sentence.append(eos_id)
            encoded_sentence = tuple(current_encoded_sentence)
            encoded_sentences.append(encoded_sentence)
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
            return None
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
        if len(decoded_corpus) == decoded_corpus.count(self._end_of_sentence_token):
            raise DecodingError("Postprocessing resulted in empty output")
        decoded_list = []
        decoded_str = ' '.join(decoded_corpus)
        decoded_str = decoded_str.replace(f' {self._end_of_sentence_token} ', '.')
        decoded_list.extend(decoded_str.split('.'))
        sentence_index = 0
        if decoded_list[-1] == '':
            decoded_list = decoded_list[:-1]
        for sentence in decoded_list:
            capitalized_sentence = sentence.capitalize()
            decoded_list[sentence_index] = capitalized_sentence
            sentence_index += 1
        decoded_text = ". ".join(decoded_list) + "."
        return decoded_text

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
        if not isinstance(text, str):
            raise EncodingError("Invalid input: text must be a string")
        if not text.strip():
            raise EncodingError("Invalid input: text must be a non-empty string")
        processed_tokens = []
        current_sentence_chars = []
        extracted_sentences = []
        for character in text:
            if character in '.!?':
                complete_sentence = ''.join(current_sentence_chars).strip()
                if complete_sentence:
                    extracted_sentences.append(complete_sentence)
                current_sentence_chars = []
            else:
                current_sentence_chars.append(character)
        if current_sentence_chars:
            final_sentence = ''.join(current_sentence_chars).strip()
            if final_sentence:
                extracted_sentences.append(final_sentence)
        for sentence in extracted_sentences:
            words = sentence.lower().split()
            clean_words = []
            for word in words:
                cleaned_word = ''
                for symbol in word:
                    if symbol.isalpha():
                        cleaned_word = cleaned_word + symbol
                if cleaned_word:
                    clean_words.append(cleaned_word)
            if clean_words:
                for clean_word in clean_words:
                    processed_tokens.append(clean_word)
                processed_tokens.append(self._end_of_sentence_token)
        if not processed_tokens:
            raise EncodingError("Tokenization resulted in empty output")
        return tuple(processed_tokens)
            
        


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
        result = []
        for child in children:
            if child.get_name() == item:
                result.append(child)
        return tuple(result)

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
        for element in prefix:
            found_child = None
            for child in current_node.get_children():
                if child.get_name() == element:
                    found_child = child
                    break
            if found_child is None:
                raise TriePrefixNotFoundError('Prefix not found in the tree')
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
            prefix_node = self.get_prefix(prefix)
        except TriePrefixNotFoundError:
            return tuple()
        if not prefix_node.has_children():
            return tuple()
        all_children_nodes = [(prefix_node, list(prefix))]
        all_sequences = []
        while all_children_nodes:
            current_node, current_sequence = all_children_nodes.pop(0)
            if not current_node.has_children():
                all_sequences.append(tuple(current_sequence))
            
            for children_node in current_node.get_children():
                all_children_nodes.append([children_node, current_sequence + [children_node.get_name()]])
        return tuple(all_sequences)

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
            for i in range(len(sentence) - self._n_gram_size + 1):
                ngram = tuple(sentence[i:i + self._n_gram_size])
                all_ngrams.append(ngram)
        try:
            for ngram in all_ngrams:
                self._insert(ngram)
            final_ngrams = self._collect_all_ngrams()
            self._fill_frequencies(final_ngrams)
            return 0
        except:
            return 1

    def get_next_tokens(self, start_sequence: NGramType) -> dict[int, float]:
        """
        Get all possible next tokens and their relative frequencies for a given prefix.

        Args:
            start_sequence (NGramType): The prefix sequence.

        Returns:
            dict[int, float]: Mapping of token → relative frequency.
        """
        child_node = self.get_prefix(start_sequence)
        if not child_node.has_children():
            return {}
        return self._collect_frequencies(child_node)

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
            generated_next_token = self.get_next_tokens(context)
        except TriePrefixNotFoundError:
            return {}
        return generated_next_token

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
        nodes_to_process = [(self._root, [])]
        while nodes_to_process:
            node, path = nodes_to_process.pop()
            node_name = node.get_name()
            if node_name is not None:
                updated_path = path + [node_name]
            else:
                updated_path = path
            if len(updated_path) == self._n_gram_size:
                all_ngrams.append(tuple(updated_path))
            else:
                children = node.get_children()
                for child in children:
                    nodes_to_process.append((child, updated_path))
        return tuple(all_ngrams)

    def _collect_frequencies(self, node: TrieNode) -> dict[int, float]:
        """
        Collect frequencies from immediate child nodes only.

        Args:
            node (TrieNode): Current node.

        Returns:
            dict[int, float]: Collected frequencies of items.
        """
        frequency_dictionary = {}
        for child_node in node.get_children():
            token = child_node.get_name()
            if not token:
                continue
            frequency = child_node.get_value()
            frequency_dictionary[token] = frequency
        return frequency_dictionary

    def _fill_frequencies(self, encoded_corpus: tuple[NGramType, ...]) -> None:
        """
        Calculate and assign frequencies for nodes in the trie based on corpus statistics.

        Counts occurrences of each n-gram and stores the relative frequency on the last node
        of each n-gram sequence.

        Args:
            encoded_corpus (tuple[NGramType, ...]): Tuple of n-grams extracted from the corpus.
        """
        n_grams = self._collect_all_ngrams()
        if not n_grams:
            return
        absolute_frequencies = {}
        for n_gram in n_grams:
            absolute_frequencies[n_gram] = absolute_frequencies.get(n_gram, 0) + 1
        all_ngrams = len(n_grams)
        for n_gram, absolute_freq in absolute_frequencies.items():
            frequency = absolute_freq / all_ngrams
            try:
                last_node = self.get_prefix(n_gram)
                last_node.set_value(frequency)
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

    def build(self) -> int:
        """
        Build N-gram tries for all possible ngrams based on a corpus of tokens.

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1.
        """
        if (
            not isinstance(self._encoded_corpus, tuple)
            or not isinstance(self._max_ngram_size, int)
            or not self._encoded_corpus
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
        if not isinstance(current_n_gram_size, int):
            raise IncorrectNgramError("The variable must be integer or None")
        if current_n_gram_size < 2 or current_n_gram_size > self._max_ngram_size:
            raise  IncorrectNgramError(
                f"N-gram size must be between 2 and {self._max_ngram_size}"
                )
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
        if self._current_n_gram_size is None or self._current_n_gram_size < 2:
            return {}
        context_length = self._current_n_gram_size - 1
        if len(sequence) < context_length:
                context = sequence
        else:
                context = sequence[-context_length:]
        try:
            prefix_node = self.get_prefix(tuple(context))
        except TriePrefixNotFoundError:
            return {}
        next_tokens = {}
        for child in prefix_node.get_children():
            name = child.get_name()
            value = child.get_value()
            if name is not None:
                next_tokens[name] = value
        return next_tokens

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
        new_node = TrieNode(name = node_name, value = freq)
        parent._children.append(new_node)
        return new_node

    def _merge(self) -> None:
        """
        Merge all built N-gram trie models into a single unified trie.
        """
        if not self._models:
            raise MergeTreesError("No models to merge")
        self._root = TrieNode(None)
        models_to_process = (
            self._models.values()
            if isinstance(self._models, dict)
            else self._models
            )
        for model in models_to_process:
            if model is None:
                continue
            model_root = model.get_root()
            if model_root is None:
                continue
            self._insert_trie(model_root)

    def _insert_trie(self, source_root: TrieNode) -> None:
        """
        Insert all nodes of source root trie into our main root.

        Args:
            source_root (TrieNode): Source root to insert tree
        """
        stack = [(source_root, self._root)]
        while stack:
            source_node, target_node = stack.pop()
            if target_node is None:
                continue
            for source_child in source_node.get_children():
                child_name = source_child.get_name()
                if child_name is not None:
                    frequency_of_child = source_child.get_value()
                    target_child = self._assign_child(
                        target_node,
                        child_name,
                        frequency_of_child
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
        if (
            not isinstance(sequence_to_continue, tuple)
            or len(sequence_to_continue) == 0
        ):
            return None
        max_ngram_size = self._dynamic_trie.get_n_gram_size()
        start_size = max_ngram_size
        end_size = 1
        step = -1
        ngram_sizes_list = list(range(start_size, end_size, step))
        for current_ngram_size in ngram_sizes_list:
            self._dynamic_trie.set_current_ngram_size(current_ngram_size)
            next_tokens_dictionary = self._dynamic_trie.generate_next_token(sequence_to_continue)
            if next_tokens_dictionary:
                return next_tokens_dictionary
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
            not isinstance(seq_len, int)
            or seq_len <= 0
            or not isinstance(prompt, str)
            or not prompt
        ):
            return None
        encoded_result = self._text_processor.encode(prompt)
        if not encoded_result:
            return None
        token_list = list(encoded_result)
        eos_token_str = self._text_processor._end_of_sentence_token
        eos_marker = self._text_processor.get_id(eos_token_str)
        if token_list and token_list[-1] == eos_marker:
            token_list = token_list[:-1]
        for _ in range(seq_len):
            candidates = self.get_next_token(tuple(token_list))
            if candidates is None or len(candidates) == 0:
                break
            selected_token, _ = max(candidates.items(),
                                key=lambda x: (x[1], x[0]))
            token_list.append(selected_token)
        word_list = []
        token_storage = getattr(self._text_processor, '_storage', {})
        for current_token_id in token_list:
            for vocab_word, vocab_id in token_storage.items():
                if vocab_id == current_token_id:
                    word_list.append(vocab_word)
                    break
        text_processor = getattr(self._text_processor, '_postprocess_decoded_text', None)
        if text_processor:
            processed_output = text_processor(tuple(word_list))
            return str(processed_output)
        return ' '.join(word_list)


def save(trie: DynamicNgramLMTrie, path: str) -> None:
    """
    Save DynamicNgramLMTrie.

    Args:
        trie (DynamicNgramLMTrie): Trie for saving
        path (str): Path for saving
    """
    root_node = trie.get_root()
    stack = [(root_node, {})]
    root_dictionary = None
    while stack:
        current_node, parent_dictionary = stack.pop()
        node_dict = {}
        if not parent_dictionary:
            node_dictionary = {
                "value": None,
                "freq": 0.0,
                "children": []
            }
            root_dictionary = node_dictionary
        else:
            node_dict = {
                "value": current_node.get_name(),
                "freq": current_node.get_value(),
                "children": []
            }
            parent_dictionary["children"].append(node_dict)
        children = current_node.get_children()
        for i in range(len(children) - 1, -1, -1):
            stack.append((children[i], node_dict))
    trie_data = {"trie": root_dictionary}
    with open(path, 'w', encoding = 'utf-8') as file:
        json.dump(trie_data, file, indent = 2)


def load(path: str) -> DynamicNgramLMTrie:
    """
    Load DynamicNgramLMTrie from file.

    Args:
        path (str): Trie path

    Returns:
        DynamicNgramLMTrie: Trie from file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        evidence = json.load(f)
    encoded_corpus = tuple(evidence.get('encoded_corpus', ()))
    max_ngram_size = evidence.get('max_ngram_size', 3)
    load_file = DynamicNgramLMTrie(encoded_corpus, max_ngram_size)
    result = load_file.build()
    if result != 0:
        return DynamicNgramLMTrie(tuple(), max_ngram_size)
    load_file.set_current_ngram_size(evidence.get('current_n_gram_size', max_ngram_size))
    return load_file
