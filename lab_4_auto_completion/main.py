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
    Exception is raised when the prefix required for the transition is not present in the tree.
    """

class EncodingError(Exception):
    """
    Exception is raised when text encoding fails due to incorrect input or a processing error.
    """

class DecodingError(Exception):
    """
    Exception is raised when text decoding fails due to incorrect input or a processing error.
    """

class IncorrectNgramError(Exception):
    """
    Exception is raised when an inappropriate n-gram size is attempted.
    """

class MergeTreesError (Exception):
    """
    Exception is raised when it is impossible to merge the trees.
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
        if not isinstance(text, str):
            raise EncodingError('Text must be a string')

        if not text.strip():
            raise EncodingError('Text cannot be empty')

        tokens = []
        without_punctuation = []
        current_sentence = ''

        for element in text:
            current_sentence += element
            if element in '?!.':
                without_punctuation.append(current_sentence.strip())
                current_sentence = ''

        if current_sentence.strip():
            without_punctuation.append(current_sentence.strip())

        for sentence in without_punctuation:
            if not sentence:
                continue

            words = sentence.lower().split()
            sentences = []

            for word in words:
                cleaned_word = ''.join(char for char in word if char.isalpha())
                if cleaned_word:
                    self._put(cleaned_word)
                    word_id = self._storage[cleaned_word]
                    sentences.append(word_id)

            id_eos = self._storage[self._end_of_sentence_token]
            sentences.append(id_eos)

            tokens.append(tuple(sentences))

        if not tokens:
            raise EncodingError('No tokens have been generated')

        return tuple(tokens)

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
        if element in self._storage:
            return

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
            raise DecodingError('Invalid input: decoded_corpus must be a non-empty tuple')

        result = []
        current_sentence = []
        for element in decoded_corpus:
            if element == self._end_of_sentence_token:
                if current_sentence:
                    result.append(' '.join(current_sentence))
                    current_sentence = []
            else:
                current_sentence.append(element)

        if current_sentence:
            result.append(' '.join(current_sentence))

        if not result:
            raise DecodingError('Postprocessing resulted in empty output')

        postprocess_text = []
        for sentence in result:
            if sentence:
                capitalaze = sentence[0].upper() + sentence[1:] + '.'
                postprocess_text.append(capitalaze)

        result = ' '.join(postprocess_text)
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
        if not isinstance(text, str):
            raise EncodingError('Invalid input: text must be a non-empty string')

        if not text.strip():
            raise EncodingError('Invalid input: text must be a non-empty string')

        tokens = []

        clear_sentences = []
        current_sentence = ""

        for char in text:
            current_sentence += char
            if char in '.!?':
                clear_sentences.append(current_sentence.strip())
                current_sentence = ""

        if current_sentence.strip():
            clear_sentences.append(current_sentence.strip())

        for sentence in clear_sentences:
            if not sentence:
                continue

            words = sentence.lower().split()
            sentence_has_valid_words = False


            for word in words:
                cleaned_word = ''.join(char for char in word if char.isalpha())
                if cleaned_word:
                    tokens.append(cleaned_word)
                    sentence_has_valid_words = True

            if sentence_has_valid_words:
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
        return "TrieNode(name={}, value={})".format(self.get_name(), self.get_value())

    def add_child(self, item: int) -> None:
        """
        Add a new child node with the given item.

        Args:
            item (int): Data value for the new child node.
        """
        if not isinstance(item, int):
            raise ValueError('Item must be an integer')

        new_node = TrieNode(item)
        self._children.append(new_node)
        return new_node

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
        children_item = tuple(x for x in self._children if x.get_name() == item)
        return children_item

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
        self._root = TrieNode(None)

    def fill(self, encoded_corpus: tuple[NGramType]) -> None:
        """
        Fill the trie based on an encoded_corpus of tokens.

        Args:
            encoded_corpus (tuple[NGramType]): Tokenized corpus.
        """

        self.clean()

        for el in encoded_corpus:
            self._insert(el)

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
            children_node = current_node.get_children()

            found_child = None

            for child in children_node:
                if child.get_name() == element:
                    found_child = child
                    break

            if found_child is None:
                raise TriePrefixNotFoundError(
                    f'Prefix {prefix} not found in trie. Failed at token: {element}'
                    )

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

        sequences = []
        stack = [(prefix_node, list(prefix))]

        while stack:
            current_node, completion = stack.pop()

            result = []
            for child in current_node.get_children():
                if child.get_name() is not None:
                    new_path = completion + [child.get_name()]

                    if not child.get_children():
                        result.append(tuple(new_path))

                    else:
                        stack.append((child, new_path))

            sequences.extend(sorted(result, reverse = True))

        return tuple(sequences)

    def _insert(self, sequence: NGramType) -> None:
        """
        Inserts a token in PrefixTrie

        Args:
            sequence (NGramType): Tokens to insert.
        """
        current_node = self._root
        for step in sequence:
            children_node = current_node.get_children()

            found_child = None

            for child in children_node:
                if child.get_name() == step:
                    found_child = child
                    break

            if found_child is None:
                new_node = current_node.add_child(step)
                current_node = new_node
            else:
                current_node = found_child


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
        self._root = TrieNode(None)

        if not self._encoded_corpus:
            return 1

        if self._n_gram_size <= 0:
            return 1

        n_grams_count = 0

        try:
            for sentence in self._encoded_corpus:
                if not sentence:
                    continue

                sentence_len = len(sentence)

                if sentence_len < self._n_gram_size:
                    continue

                for i in range(len(sentence) - self._n_gram_size + 1):
                    n_gram = tuple(sentence[i:i + self._n_gram_size])
                    for prefix_len in range(1, len(n_gram) + 1):
                        prefix = n_gram[:prefix_len]
                        self._insert(prefix)

                    n_grams_count += 1

            if n_grams_count == 0:
                return 1

            all_ngrams = self._collect_all_ngrams()
            if all_ngrams:
                self._fill_frequencies(all_ngrams)
            else:
                return

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
        try:
            node = self.get_prefix(start_sequence)
        except TriePrefixNotFoundError:
            raise

        if not node.get_children():
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
        if len(sequence) < self.get_n_gram_size() - 1:
            return None
        context = sequence[-(self.get_n_gram_size() - 1):]

        try:
            generated_tokens = self.get_next_tokens(context)
            return generated_tokens

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
        if not isinstance(new_corpus, tuple) or not new_corpus:
            return None
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
        n_grams = []
        stack = []
        initial_sequence = []
        stack.append((self._root, initial_sequence))

        while stack:
            node, current_sequence = stack.pop()

            if len(current_sequence) == self.get_n_gram_size():
                n_grams.append(tuple(current_sequence))
                continue

            for child in node.get_children():
                if child.get_name() is not None:
                    new_sequence = current_sequence + [child.get_name()]
                    stack.append((child, new_sequence))

        return tuple(n_grams)


    def _collect_frequencies(self, node: TrieNode) -> dict[int, float]:
        """
        Collect frequencies from immediate child nodes only.

        Args:
            node (TrieNode): Current node.

        Returns:
            dict[int, float]: Collected frequencies of items.
        """
        dictionary = {}
        for element in node.get_children():
            token = element.get_name()
            if token is not None:
                freq_token = element.get_value()
                dictionary[token] = freq_token

        return dictionary

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
        NGramTrieLanguageModel.__init__(self, encoded_corpus, n_gram_size)
        self._root = TrieNode()
        self._encoded_corpus = encoded_corpus
        self._max_ngram_size = n_gram_size
        self._models = {}
        self._current_n_gram_size = 0

    def get_root(self) -> TrieNode:
        """
        Get the root node of the trie.

        Returns:
           TrieNode: The root node of the prefix trie
        """
        return self._root

    def build(self) -> int:
        """
        Build N-gram tries for all possible ngrams based on a corpus of tokens.

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1.
        """
        if (not isinstance(self._max_ngram_size, int) or
            self._max_ngram_size < 2 or
            not isinstance(self._encoded_corpus, tuple) or
            not self._encoded_corpus):
            return 1

        for sentence in self._encoded_corpus:
            if not isinstance(sentence, tuple):
                return 1

            for token in sentence:
                if not isinstance(token, int):
                    return 1

        max_length = 0
        for element in self._encoded_corpus:
            if element:
                max_length = max(max_length, len(element))

        if max_length < 2:
            return 1

        max_ngram_to_build = min(max_length, self._max_ngram_size)
        if max_ngram_to_build < 2:
            return 1

        self._models = {}

        for n_gram_size in range(2, max_ngram_to_build + 1):
            model = NGramTrieLanguageModel(self._encoded_corpus, n_gram_size)
            if model.build() == 0:
                self._models[n_gram_size] = model
            else:
                return 1

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
            self._current_n_gram_size = None
            return

        if not isinstance(current_n_gram_size, int):
            raise IncorrectNgramError('The variable must be an integer or None')

        if current_n_gram_size < 2 or current_n_gram_size > self._max_ngram_size:
            raise  IncorrectNgramError(
                f'N-gram size must be between 2 and {self._max_ngram_size}'
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

        context_length = min(self._current_n_gram_size - 1, len(sequence))
        if len(sequence) < self._current_n_gram_size - 1:
            return self._collect_frequencies(self._root) if self._root else {}

        current_node = self._root
        if context_length > 0:
            context = sequence[-context_length:]

            for element in context:
                found_child = None

                for child in current_node.get_children():
                    if child.get_name() == element:
                        found_child = child
                        break

                if found_child is None:
                    return {}

                current_node = found_child

        next_tokens = {}
        for child in current_node.get_children():
            name = child.get_name()
            value = child.get_value()

            if name is not None and value is not None:
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
        if not isinstance(node_name, int) or node_name < 0:
            raise ValueError('The parameter must be an integer and must be positive')

        for child in parent.get_children():
            if child.get_name() == node_name:
                if freq != 0.0:
                    child.set_value(freq)
                return child

        new_children = TrieNode(node_name)
        parent.add_child(node_name)

        for child in parent.get_children():
            if child.get_name() == node_name:
                if freq != 0.0:
                    child.set_value(freq)
                return child

        return new_children

    def _merge(self) -> None:
        """
        Merge all built N-gram trie models into a single unified trie.
        """
        if not self._models:
            raise MergeTreesError("No models to merge")

        self._root = TrieNode(None)

        if isinstance(self._models, dict):
            for model in self._models.values():
                if model is not None:
                    model_root = model.get_root()
                    if model_root is not None:
                        self._insert_trie(model_root)

        else:
            for element in self._models:
                if element is not None:
                    model_root = element.get_root()
                if model_root is not None:
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
        if not isinstance(sequence_to_continue, tuple) or not sequence_to_continue:
            return None

        max_n_gram_size = self._dynamic_trie._max_ngram_size
        max_use_n = min(max_n_gram_size, len(sequence_to_continue) + 1)

        n_gram_size = list(range(max_use_n, 1, -1))
        if not n_gram_size:
            return None

        for ngram_size in n_gram_size:
            try:
                self._dynamic_trie.set_current_ngram_size(ngram_size)
                candidates = self._dynamic_trie.generate_next_token(sequence_to_continue)
                if candidates is None:
                    continue

                if candidates:
                    return candidates

            except Exception:
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
        if (
            not isinstance(seq_len, int) or seq_len <= 0
            or not isinstance(prompt, str) or not prompt.strip()
            or self._text_processor.encode(prompt) is None
        ):
            return None

        try:
            tokenized_text = self._text_processor._tokenize(prompt)
            if tokenized_text is None:
                return None
        except Exception:
            return None

        eos_char = self._text_processor._end_of_sentence_token
        token_list = list(tokenized_text)
        if token_list and token_list[-1] == eos_char:
            token_list.pop()

        encoded_prompt = []
        for element in token_list:
            self._text_processor._put(element)
            encoded_prompt.append(self._text_processor._storage[element])

        if not encoded_prompt:
            return None

        seq_list = list(encoded_prompt)
        for _ in range(seq_len):
            context = tuple(seq_list)
            tokens = self.get_next_token(context)
            if tokens is None or not tokens:
                break

            best_candidates = sorted(tokens.items(),
                                key=lambda x: (-x[1], -x[0]))[0][0]
            seq_list.append(best_candidates)

        decoded_words = []
        for token_id in seq_list:
            for word, word_id in self._text_processor._storage.items():
                if word_id == token_id:
                    decoded_words.append(word)
                    break

        try:
            postprocess_method = self._text_processor._postprocess_decoded_text
            return str(postprocess_method(tuple(decoded_words)))
        except AttributeError:
            return ' '.join(decoded_words)

def save(trie: DynamicNgramLMTrie, path: str) -> None:
    """
    Save DynamicNgramLMTrie.

    Args:
        trie (DynamicNgramLMTrie): Trie for saving
        path (str): Path for saving
    """
    root_node = trie.get_root()

    stack = [(root_node, None)]
    root_dict = None

    while stack:
        current_node, parent_dictionary = stack.pop()
        if parent_dictionary is None:
            node_dict = {
                "value": None,
                "freq": 0.0,
                "children": []
            }
            root_dict = node_dict
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

    trie_data = {"trie": root_dict}
    with open(path, 'w', encoding = 'utf-8') as f:
        json.dump(trie_data, f, indent = 2)


def load(path: str) -> DynamicNgramLMTrie:
    """
    Load DynamicNgramLMTrie from file.

    Args:
        path (str): Trie path

    Returns:
        DynamicNgramLMTrie: Trie from file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    information = data.get("trie", {})
    trie = DynamicNgramLMTrie((), 3)
    if not information:
        return trie

    nodes = [(information, None)]

    while nodes:
        node_data, parent_node = nodes.pop()
        value = node_data.get("value")
        freq = node_data.get("freq", 0.0)
        current_node = TrieNode(value, freq)

        if parent_node is None:
            trie._root = current_node
        else:
            parent_node._children.append(current_node)

        children_data = node_data.get("children", [])
        for child_data in children_data[::-1]:
            nodes.append((child_data, current_node))

    return trie
