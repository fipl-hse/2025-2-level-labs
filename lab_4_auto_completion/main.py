"""
Lab 4
"""


import string
import json

# pylint: disable=unused-argument, super-init-not-called, unused-private-member, duplicate-code, unused-import


#from lab_3_generate_by_ngrams.main import BackOffGenerator, NGramLanguageModel, TextProcessor

class TextProcessor:
    """
    Handle text tokenization, encoding and decoding.

    Attributes:
        _end_of_word_token (str): A token denoting word boundary
        _storage (dict): Dictionary in the form of <token: identifier>
    """

    def __init__(self, end_of_word_token: str) -> None:
        """
        Initialize an instance of LetterStorage.

        Args:
            end_of_word_token (str): A token denoting word boundary
        """
        self._end_of_word_token = end_of_word_token
        self._storage = {end_of_word_token: 0}

    def _tokenize(self, text: str) -> tuple[str, ...] | None:
        """
        Tokenize text into unigrams, separating words with special token.

        Punctuation and digits are removed. EoW token is appended after the last word in two cases:
        1. It is followed by punctuation
        2. It is followed by space symbol

        Args:
            text (str): Original text

        Returns:
            tuple[str, ...] | None: Tokenized text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(text, str):
            return None
        tokens = []
        special_symbols = set(string.punctuation)
        special_symbols.remove("-")
        for token in text.lower():
            if token.isalpha():
                tokens.append(token)
            elif token.isspace() or token in special_symbols:
                if tokens[-1] != self._end_of_word_token:
                    tokens.append(self._end_of_word_token)
                else:
                    continue
            elif token.isdigit():
                continue
        return tuple(tokens) if tokens else None


    def get_id(self, element: str) -> int | None:
        """
        Retrieve a unique identifier of an element.

        Args:
            element (str): String element to retrieve identifier for

        Returns:
            int | None: Integer identifier that corresponds to the given element

        In case of corrupt input arguments or arguments not included in storage,
        None is returned
        """
        if not isinstance(element, str) or element not in self._storage:
            return None
        return self._storage.get(element)

    def get_end_of_word_token(self) -> str:  # type: ignore[empty-body]
        """
        Retrieve value stored in self._end_of_word_token attribute.

        Returns:
            str: EoW token
        """
        return self._end_of_word_token

    def get_token(self, element_id: int) -> str | None:
        """
        Retrieve an element by unique identifier.

        Args:
            element_id (int): Identifier to retrieve identifier for

        Returns:
            str | None: Element that corresponds to the given identifier

        In case of corrupt input arguments or arguments not included in storage, None is returned
        """
        if not isinstance(element_id, int):
            return None
        result = next((key for key, value in self._storage.items() if value == element_id), None)
        return result

    def encode(self, text: str) -> tuple[int, ...] | None:
        """
        Encode text.

        Tokenize text, assign each symbol an integer identifier and
        replace letters with their ids.

        Args:
            text (str): An original text to be encoded

        Returns:
            tuple[int, ...] | None: Processed text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(text, str) or text == "":
            return None
        tokenized_text = self._tokenize(text)
        if tokenized_text is not None:
            list_with_id = []
            for each_element in tokenized_text:
                self._put(each_element)
                value_id = self.get_id(each_element)
                if value_id is None:
                    return None
                list_with_id.append(value_id)
        else:
            return None
        return tuple(list_with_id)

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): An element to put into storage

        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """
        if not isinstance(element, str) or len(element) != 1:
            return None
        if element not in self._storage:
            self._storage[element] = len(self._storage)
        return None


    def decode(self, encoded_corpus: tuple[int, ...]) -> str | None:
        """
        Decode and postprocess encoded corpus by converting integer identifiers to string.

        Special symbols are replaced with spaces (no multiple spaces in a row are allowed).
        The first letter is capitalized, resulting sequence must end with a full stop.

        Args:
            encoded_corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            str | None: Resulting text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(encoded_corpus, tuple) or len(encoded_corpus) == 0:
            return None
        decoded_text = self._decode(encoded_corpus)
        if decoded_text is None:
            return None
        postprocess_decoded_text = self._postprocess_decoded_text(decoded_text)
        return postprocess_decoded_text

    def fill_from_ngrams(self, content: dict) -> None:
        """
        Fill internal storage with letters from external JSON.

        Args:
            content (dict): ngrams from external JSON
        """
        if not isinstance(content, dict):
            return None
        if not content:
            return None
        for el in content["freq"]:
            for symbol in el.lower():
                if symbol.isalpha():
                    self._put(symbol)
        return None

    def _decode(self, corpus: tuple[int, ...]) -> tuple[str, ...] | None:
        """
        Decode sentence by replacing ids with corresponding letters.

        Args:
            corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            tuple[str, ...] | None: Sequence with decoded tokens

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(corpus, tuple) or len(corpus) == 0:
            return None
        result = ""
        for element in corpus:
            symbol = self.get_token(element)
            if symbol is None:
                return None
            result += symbol
        return tuple(result)

    def _postprocess_decoded_text(self, decoded_corpus: tuple[str, ...]) -> str | None:
        """
        Convert decoded sentence into the string sequence.

        Special symbols are replaced with spaces (no multiple spaces in a row are allowed).
        The first letter is capitalized, resulting sequence must end with a full stop.

        Args:
            decoded_corpus (tuple[str, ...]): A tuple of decoded tokens

        Returns:
            str | None: Resulting text

        In case of corrupt input arguments, None is returned
        """
        if not isinstance(decoded_corpus, tuple) or not decoded_corpus:
            return None
        result = ""
        for element in decoded_corpus:
            if element == self._end_of_word_token:
                result += " "
            else:
                result += element
        result_without_ = result.strip()
        result = result_without_.rstrip('.')
        return result.capitalize() + "."


class NGramLanguageModel:
    """
    Store language model by n_grams, predict the next token.

    Attributes:
        _n_gram_size (int): A size of n-grams to use for language modelling
        _n_gram_frequencies (dict): Frequencies for n-grams
        _encoded_corpus (tuple): Encoded text
    """

    def __init__(self, encoded_corpus: tuple | None, n_gram_size: int) -> None:
        """
        Initialize an instance of NGramLanguageModel.

        Args:
            encoded_corpus (tuple | None): Encoded text
            n_gram_size (int): A size of n-grams to use for language modelling
        """
        self._encoded_corpus = encoded_corpus
        self._n_gram_size = n_gram_size
        self._n_gram_frequencies = {}

    def get_n_gram_size(self) -> int:  # type: ignore[empty-body]
        """
        Retrieve value stored in self._n_gram_size attribute.

        Returns:
            int: Size of stored n_grams
        """
        return self._n_gram_size


    def set_n_grams(self, frequencies: dict) -> None:
        """
        Setter method for n-gram frequencies.

        Args:
            frequencies (dict): Computed in advance frequencies for n-grams
        """
        if not isinstance(frequencies, dict):
            return None
        if not frequencies:
            return None
        self._n_gram_frequencies = frequencies
        return None

    def build(self) -> int:  # type: ignore[empty-body]
        """
        Fill attribute `_n_gram_frequencies` from encoded corpus.

        Encoded corpus is stored in the attribute `_encoded_corpus`

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1

        In case of corrupt input arguments or methods used return None,
        1 is returned
        """
        if not isinstance(self._encoded_corpus, tuple) or not self._encoded_corpus:
            return 1
        current_encoded_corpus = self._extract_n_grams(self._encoded_corpus)
        if not current_encoded_corpus:
            return 1
        n_gram_counts = {}
        prefix_counts = {}
        for n_gram in current_encoded_corpus:
            n_gram_counts[n_gram] = n_gram_counts.get(n_gram, 0) + 1
            context = n_gram[:-1]
            prefix_counts[context] = prefix_counts.get(context, 0) + 1
        for n_gram, count in n_gram_counts.items():
            context = n_gram[:-1]
            self._n_gram_frequencies[n_gram] = count / prefix_counts[context]
        return 0

    def generate_next_token(self, sequence: tuple[int, ...]) -> dict | None:
        """
        Retrieve tokens that can continue the given sequence along with their probabilities.

        Args:
            sequence (tuple[int, ...]): A sequence to match beginning of NGrams for continuation

        Returns:
            dict | None: Possible next tokens with their probabilities

        In case of corrupt input arguments, None is returned
        """
        if (not isinstance(sequence, tuple)
            or not sequence
            or len(sequence) < (self._n_gram_size - 1)):
            return None
        result = {}
        context = sequence[-(self._n_gram_size - 1):]
        for element in self._n_gram_frequencies:
            if element[:self._n_gram_size - 1] == context:
                result[element[-1]] = self._n_gram_frequencies.get(element)
        sorted_result = dict(sorted(result.items(), key=lambda x: (x[1], x[0]), reverse=True))
        return sorted_result

    def _extract_n_grams(
        self, encoded_corpus: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...] | None:
        """
        Split encoded sequence into n-grams.

        Args:
            encoded_corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            tuple[tuple[int, ...], ...] | None: A tuple of extracted n-grams

        In case of corrupt input arguments, None is returned
        """
        if not isinstance(encoded_corpus, tuple) or not encoded_corpus:
            return None
        result = []
        for i in range(len(encoded_corpus) - self._n_gram_size + 1):
            n_gram = encoded_corpus[i:i + self._n_gram_size]
            result.append(tuple(n_gram))
        return tuple(result)

class BackOffGenerator:
    """
    Language model for back-off based text generation.

    Attributes:
        _language_models (dict[int, NGramLanguageModel]): Language models for next token prediction
        _text_processor (NGramLanguageModel): A TextProcessor instance to handle text processing
    """

    def __init__(
        self, language_models: tuple[NGramLanguageModel, ...], text_processor: TextProcessor
    ) -> None:
        """
        Initializes an instance of BackOffGenerator.

        Args:
            language_models (tuple[NGramLanguageModel, ...]):
                Language models to use for text generation
            text_processor (TextProcessor): A TextProcessor instance to handle text processing
        """
        self._language_models = {}
        for language_model in language_models:
            self._language_models[language_model.get_n_gram_size()] = language_model
        self._text_processor = text_processor

    def run(self, seq_len: int, prompt: str) -> str | None:
        """
        Generate sequence based on NGram language model and prompt provided.

        Args:
            seq_len (int): Number of tokens to generate
            prompt (str): Beginning of sequence

        Returns:
            str | None: Generated sequence

        In case of corrupt input arguments or methods used return None,
        None is returned
        """
        if not isinstance(seq_len, int) or not isinstance(prompt, str):
            return None
        if not prompt:
            return None
        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None
        list_sequence = list(encoded_prompt)
        for _ in range(seq_len):
            candidates = self._get_next_token(tuple(list_sequence))
            if not candidates:
                break
            next_element = max(candidates.items(), key=lambda x: x[1])[0]
            list_sequence.append(next_element)
        return self._text_processor.decode(tuple(list_sequence))

    def _get_next_token(self, sequence_to_continue: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            dict[int, float] | None: Next tokens for sequence continuation

        In case of corrupt input arguments return None.
        """
        if not isinstance(sequence_to_continue, tuple):
            return None
        if not sequence_to_continue:
            return None
        sizes_of_n_grams = [model.get_n_gram_size() for model in self._language_models.values()]
        sizes_of_n_grams.sort(reverse=True)
        for size_of_n_gram in sizes_of_n_grams:
            language_model = self._language_models[size_of_n_gram]
            candidates = language_model.generate_next_token(sequence_to_continue)
            if candidates is not None and candidates:
                return candidates
        return None


NGramType = tuple[int, ...]
"Type alias for NGram."

class TriePrefixNotFoundError(Exception):
    """Exception raised when required prefix is not found in the trie."""


class EncodingError(Exception):
    """Exception raised when text encoding fails due to incorrect input or processing error."""


class DecodingError(Exception):
    """Exception raised when text decoding fails due to incorrect input or processing error."""


class IncorrectNgramError(Exception):
    """Exception raised when attempting to use inappropriate n-gram size."""


class MergeTreesError(Exception):
    """Exception raised when it's impossible to merge trees."""

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
        if not isinstance(text, str) or not text:
            raise EncodingError("Invalid input text")
        tokenized = self._tokenize(text)
        if not tokenized:
            raise EncodingError("Failed to tokenize text")
        sentences = []
        current_sentence = []
        for token in tokenized:
            if token == self._end_of_sentence_token:
                if current_sentence:
                    encoded_sentence = []
                    for word in current_sentence:
                        self._put(word)
                        word_id = self.get_id(word)
                        if word_id is None:
                            raise EncodingError(f"Failed to encode word: {word}")
                        encoded_sentence.append(word_id)
                    encoded_sentence.append(0)
                    sentences.append(tuple(encoded_sentence))
                    current_sentence = []
            else:
                current_sentence.append(token)
        if current_sentence:
            encoded_sentence = []
            for word in current_sentence:
                self._put(word)
                word_id = self.get_id(word)
                if word_id is None:
                    raise EncodingError(f"Failed to encode word: {word}")
                encoded_sentence.append(word_id)
            encoded_sentence.append(0)
            sentences.append(tuple(encoded_sentence))
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
        filtered_corpus = [token for token in decoded_corpus
                           if token != self._end_of_sentence_token]
        if not filtered_corpus:
            raise DecodingError("Postprocessing resulted in empty output")
        sentences = []
        current_sentence = []
        for element in decoded_corpus:
            if element == self._end_of_sentence_token:
                if current_sentence:
                    sentences.append(" ".join(current_sentence))
                    current_sentence = []
            else:
                current_sentence.append(element)
        if current_sentence:
            sentences.append(" ".join(current_sentence))
        if not sentences:
            raise DecodingError("Postprocessing resulted in empty output")
        processed_sentences = []
        for sentence in sentences:
            if sentence:
                processed = (sentence[0].upper() + sentence[1:]
                             if len(sentence) > 1 else sentence.upper())
                processed_sentences.append(processed)
        result = ". ".join(processed_sentences)
        if not result.endswith('.'):
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
        child_node = TrieNode(item)
        self._children.append(child_node)

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
        for child in self._children:
            if child.get_name() == item:
                return (child,)
        return tuple()

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
        return len(self._children) > 0


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
            children = current_node.get_children(item)
            if not children:
                raise TriePrefixNotFoundError("Prefix not found in trie")
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
        initial_seq = tuple(prefix)
        queue = [(prefix_node, initial_seq)]
        while queue:
            current_node, current_sequence = queue.pop()
            if current_node.has_children():
                children = list(current_node.get_children())
                for child in children[::-1]:
                    child_name = child.get_name()
                    if child_name is None:
                        continue
                    new_sequence = current_sequence + (child_name,)
                    sequences.append(new_sequence)
                    queue.append((child, new_sequence))
        return tuple(sequences[::-1])

    def _insert(self, sequence: NGramType) -> None:
        """
        Inserts a token in PrefixTrie

        Args:
            sequence (NGramType): Tokens to insert.
        """
        current_node = self._root
        for item in sequence:
            children = current_node.get_children(item)
            if children:
                current_node = children[0]
            else:
                current_node.add_child(item)
                children = current_node.get_children(item)
                if children:
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
        PrefixTrie.__init__(self)
        NGramLanguageModel.__init__(self, encoded_corpus, n_gram_size)
        self._n_gram_size = n_gram_size
        self._n_gram_frequencies = {}

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
        except TriePrefixNotFoundError:
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
        if (
            not current_n_gram_size
            or not isinstance(current_n_gram_size, int)
            or current_n_gram_size < 2
            or self._max_ngram_size < current_n_gram_size
        ):
            raise IncorrectNgramError('Error! Incorrect input data')
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
        return None

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
        if not source_root:
            return
        stack = [(source_root, self._root)]
        while stack:
            source_node, dest_node = stack.pop()
            children = source_node.get_children()
            for source_child in children:
                child_name = source_child.get_name()
                if child_name is not None:
                    dest_child = self._assign_child(dest_node, child_name,
                                                  source_child.get_value())
                    stack.append((source_child, dest_child))

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
            not seq_len
            or not isinstance(seq_len, int)
            or seq_len <= 0
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
