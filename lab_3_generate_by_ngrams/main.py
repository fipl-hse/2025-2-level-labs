"""
Lab 3.

Beam-search and natural language generation evaluation
"""

# pylint:disable=too-few-public-methods, unused-import
import json
import math

from lab_1_keywords_tfidf.main import check_positive_int


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
        self._storage = {self._end_of_word_token: 0}

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
        if not isinstance(text, str) or not text:
            return None
        words = text.lower().split()
        tokens = []
        for word in words:
            word_tokens = []
            for char in word:
                if char.isalpha():
                    word_tokens.append(char)
            if word_tokens:
                tokens.extend(word_tokens)
                tokens.append(self._end_of_word_token)
        if tokens and text[-1].isalnum():
            tokens.pop()
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
        if not isinstance(element, str):
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
        for element, value in self._storage.items():
            if value == element_id:
                return element
        return None

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
        if not isinstance(text, str) or not text:
            return None
        tokenized_text = self._tokenize(text) or []
        if not tokenized_text:
            return None
        encoded_corpus = []
        for token in tokenized_text:
            self._put(token)
            token_id = self.get_id(token)
            if token_id is None:
                return None
            encoded_corpus.append(token_id)
        return tuple(encoded_corpus)

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): An element to put into storage

        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """
        if (
            not isinstance(element, str)
                    or not element
                    or len(element) != 1
            ):
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
        if not isinstance(encoded_corpus, tuple) or not encoded_corpus:
            return None
        encoded_result = self._decode(encoded_corpus)
        if encoded_result is None:
            return None
        result = self._postprocess_decoded_text(encoded_result)
        if result is None:
            return None
        return str(result)

    def fill_from_ngrams(self, content: dict) -> None:
        """
        Fill internal storage with letters from external JSON.

        Args:
            content (dict): ngrams from external JSON
        """
        if not isinstance(content, dict) or not content:
            return
        for n_gram in content['freq']:
            for char in n_gram.lower():
                if char.isalpha():
                    self._put(char)

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
        if not isinstance(corpus, tuple) or not corpus:
            return None
        decoded_corpus = []
        for element_id in corpus:
            element = self.get_token(element_id)
            if element is None:
                return None
            decoded_corpus.append(element)
        return tuple(decoded_corpus)

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
        decoded_text = ''.join(decoded_corpus).replace('_', ' ').capitalize()
        if decoded_text[-1] == ' ':
            return decoded_text[:-1] + '.'
        return decoded_text + '.'


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
        if not isinstance(frequencies, dict) or not frequencies:
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
        n_grams = self._extract_n_grams(self._encoded_corpus)
        if not isinstance(n_grams, tuple) or not n_grams:
            return 1
        n_gram_counts = {}
        prefix_counts = {}
        for n_gram in n_grams:
            n_gram_counts[n_gram] = n_gram_counts.get(n_gram, 0) + 1
            prefix_counts[n_gram[:-1]] = prefix_counts.get(n_gram[:-1], 0) + 1
        for n_gram, amount in n_gram_counts.items():
            prefix_count = prefix_counts.get(n_gram[:-1], 0)
            if prefix_count == 0:
                return 1
            self._n_gram_frequencies[n_gram] = amount / prefix_count
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
        if (
            not isinstance(sequence, tuple)
            or not sequence
            or len(sequence) < self._n_gram_size - 1
            ):
            return None
        context = sequence[-(self._n_gram_size - 1):]
        result = {}
        for n_gram, freq in self._n_gram_frequencies.items():
            if n_gram[:-1] == context:
                token = n_gram[-1]
                if token not in result:
                    result[token] = freq
        return result

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
        n_gram_size = self._n_gram_size
        corpus_len = len(encoded_corpus)
        if corpus_len < n_gram_size:
            return tuple()
        n_grams = (encoded_corpus[i:i + n_gram_size]
                   for i in range(corpus_len - n_gram_size + 1))
        return tuple(n_grams)


class GreedyTextGenerator:
    """
    Greedy text generation by N-grams.

    Attributes:
        _model (NGramLanguageModel): A language model to use for text generation
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
    """

    def __init__(self, language_model: NGramLanguageModel, text_processor: TextProcessor) -> None:
        """
        Initialize an instance of GreedyTextGenerator.

        Args:
            language_model (NGramLanguageModel): A language model to use for text generation
            text_processor (TextProcessor): A TextProcessor instance to handle text processing
        """
        self._model = language_model
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
        if (
            not isinstance(seq_len, int)
            or not isinstance(prompt, str)
            or not prompt
            ):
            return None
        seq = self._text_processor.encode(prompt)
        n_gram_size = self._model.get_n_gram_size()
        if not seq or not n_gram_size:
            return None
        for _ in range(seq_len):
            next_tokens = self._model.generate_next_token(seq[-(n_gram_size + 1):])
            if not next_tokens:
                return self._text_processor.decode(seq)
            next_token = sorted(next_tokens.items(),
                                key=lambda item: (item[1], item[0]),
                                reverse=True)[0][0]
            seq += (next_token,)
        return self._text_processor.decode(seq)


class BeamSearcher:
    """
    Beam Search algorithm for diverse text generation.

    Attributes:
        _beam_width (int): Number of candidates to consider at each step
        _model (NGramLanguageModel): A language model to use for next token prediction
    """

    def __init__(self, beam_width: int, language_model: NGramLanguageModel) -> None:
        """
        Initialize an instance of BeamSearchAlgorithm.

        Args:
            beam_width (int): Number of candidates to consider at each step
            language_model (NGramLanguageModel): A language model to use for next token prediction
        """
        self._beam_width = beam_width
        self._model = language_model

    def get_next_token(self, sequence: tuple[int, ...]) -> list[tuple[int, float]] | None:
        """
        Retrieve candidate tokens for sequence continuation.

        The valid candidate tokens are those that are included in the N-gram with.
        Number of tokens retrieved must not be bigger that beam width parameter.

        The return value has the following format: [(token, probability), ...].
        The return value length matches the Beam Size parameter.

        Args:
            sequence (tuple[int, ...]): Base sequence to continue

        Returns:
            list[tuple[int, float]] | None: Tokens to use for base sequence continuation

        In case of corrupt input arguments or methods used return None.
        """
        if not isinstance(sequence, tuple) or not sequence:
            return None
        result = self._model.generate_next_token(sequence)
        if result is None:
            return None
        if not result:
            return []
        return sorted([(token, float(freq)) for token, freq in result.items()],
                      key=lambda item: item[1], reverse=True)[:self._beam_width]

    def continue_sequence(
        self,
        sequence: tuple[int, ...],
        next_tokens: list[tuple[int, float]],
        sequence_candidates: dict[tuple[int, ...], float],
    ) -> dict[tuple[int, ...], float] | None:
        """
        Generate new sequences from the base sequence with next tokens provided.

        The base sequence is deleted after continued variations are added.

        Args:
            sequence (tuple[int, ...]): Base sequence to continue
            next_tokens (list[tuple[int, float]]): Token for sequence continuation
            sequence_candidates (dict[tuple[int, ...], float]):
                Storage with all sequences generated

        Returns:
            dict[tuple[int, ...], float] | None: Updated sequence candidates

        In case of corrupt input arguments or unexpected behaviour of methods used return None.
        """
        if (
            not isinstance(sequence, tuple)
            or not isinstance(next_tokens, list)
            or not isinstance(sequence_candidates, dict)
            or not len(next_tokens) <= self._beam_width
            or not sequence in sequence_candidates
            ):
            return None
        if (
            not sequence
            or not next_tokens
            or not sequence_candidates
        ):
            return None
        copy_seq_candidates = sequence_candidates.copy()
        for token in next_tokens:
            new_seq = sequence + (token[0],)
            new_freq = copy_seq_candidates[sequence] - math.log(token[1])
            copy_seq_candidates[new_seq] = new_freq
        del copy_seq_candidates[sequence]
        return copy_seq_candidates

    def prune_sequence_candidates(
        self, sequence_candidates: dict[tuple[int, ...], float]
    ) -> dict[tuple[int, ...], float] | None:
        """
        Remove those sequence candidates that do not make top-N most probable sequences.

        Args:
            sequence_candidates (dict[tuple[int, ...], float]): Current candidate sequences

        Returns:
            dict[tuple[int, ...], float] | None: Pruned sequences

        In case of corrupt input arguments return None.
        """
        if (not isinstance(sequence_candidates, dict) or not sequence_candidates):
            return None
        return dict(sorted(sequence_candidates.items(),
                      key=lambda item: item[1], reverse=False)[:self._beam_width])


class BeamSearchTextGenerator:
    """
    Class for text generation with BeamSearch.

    Attributes:
        _language_model (tuple[NGramLanguageModel]): Language models for next token prediction
        _text_processor (NGramLanguageModel): A TextProcessor instance to handle text processing
        _beam_width (NGramLanguageModel): Beam width parameter for generation
        beam_searcher (NGramLanguageModel): Searcher instances for each language model
    """

    def __init__(
        self, language_model: NGramLanguageModel, text_processor: TextProcessor, beam_width: int
    ) -> None:
        """
        Initializes an instance of BeamSearchTextGenerator.

        Args:
            language_model (NGramLanguageModel): Language model to use for text generation
            text_processor (TextProcessor): A TextProcessor instance to handle text processing
            beam_width (int): Beam width parameter for generation
        """
        self._text_processor = text_processor
        self._beam_width = beam_width
        self.beam_searcher = BeamSearcher(self._beam_width, language_model)
        self._language_model = language_model

    def run(self, prompt: str, seq_len: int) -> str | None:
        """
        Generate sequence based on NGram language model and prompt provided.

        Args:
            prompt (str): Beginning of sequence
            seq_len (int): Number of tokens to generate

        Returns:
            str | None: Generated sequence

        In case of corrupt input arguments or methods used return None,
        None is returned
        """
        if (
            not isinstance(prompt, str)
            or not prompt
            or not check_positive_int(seq_len)
            or not seq_len
            ):
            return None
        encoded_prompt = self._text_processor.encode(prompt)
        if not encoded_prompt:
            return None
        sequence_candidates = {encoded_prompt: 0.0}
        for _ in range(seq_len):
            for sequence in sequence_candidates:
                next_tokens = self._get_next_token(sequence)
                if next_tokens is None:
                    return None
                continued = (self.beam_searcher.continue_sequence
                             (sequence, next_tokens, sequence_candidates))
                if continued is None:
                    continue
                pruned_candidates = self.beam_searcher.prune_sequence_candidates(continued)
                if pruned_candidates is None:
                    return None
                sequence_candidates = pruned_candidates
            if not sequence_candidates:
                return None
        result = self._text_processor.decode(min(sequence_candidates.items(),
                                                 key=lambda x: x[1])[0])
        return result

    def _get_next_token(
        self, sequence_to_continue: tuple[int, ...]
    ) -> list[tuple[int, float]] | None:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            list[tuple[int, float]] | None: Next tokens for sequence
            continuation

        In case of corrupt input arguments return None.
        """
        if not isinstance(sequence_to_continue, tuple) or not sequence_to_continue:
            return None
        return self.beam_searcher.get_next_token(sequence_to_continue)


class NGramLanguageModelReader:
    """
    Factory for loading language models ngrams from external JSON.

    Attributes:
        _json_path (str): Local path to assets file
        _eow_token (str): Special token for text processor
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
    """

    def __init__(self, json_path: str, eow_token: str) -> None:
        """
        Initialize reader instance.

        Args:
            json_path (str): Local path to assets file
            eow_token (str): Special token for text processor
        """
        self._json_path = json_path
        self._eow_token = eow_token
        with open(self._json_path, 'r', encoding="utf-8") as file:
            self._content = json.load(file)
        self._text_processor = TextProcessor(self._eow_token)
        self._text_processor.fill_from_ngrams(self._content)

    def load(self, n_gram_size: int) -> NGramLanguageModel | None:
        """
        Fill attribute `_n_gram_frequencies` from dictionary with N-grams.

        The N-grams taken from dictionary must be cleaned from digits and punctuation,
        their length must match n_gram_size, and spaces must be replaced with EoW token.

        Args:
            n_gram_size (int): Size of ngram

        Returns:
            NGramLanguageModel | None: Built language model.

        In case of corrupt input arguments or unexpected behaviour of methods used, return 1.
        """
        if (
            not isinstance(n_gram_size, int)
            or not n_gram_size
            or n_gram_size < 2
            ):
            return None
        n_grams = {}
        for ngram in self._content['freq']:
            processed_chars = []
            for char in ngram:
                if char.isspace():
                    processed_chars.append(0)
                elif char.isalpha():
                    char_id = self._text_processor.get_id(char.lower())
                    if char_id is not None:
                        processed_chars.append(char_id)
            if processed_chars:
                n_grams[tuple(processed_chars)] = (n_grams.get(tuple(processed_chars), 0.0) +
                                                   self._content['freq'][ngram])
        context_frequencies = {}
        for ngram, freq in n_grams.items():
            if len(ngram) == n_gram_size:
                context = ngram[:-1]
                context_frequencies[context] = context_frequencies.get(context, 0) + freq
        conditional_probabilities = {}
        for ngram, freq in n_grams.items():
            if len(ngram) == n_gram_size:
                context = ngram[:-1]
                context_freq = context_frequencies.get(context, 0)
                if context_freq > 0:
                    conditional_probabilities[ngram] = freq / context_freq
        model = NGramLanguageModel(None, n_gram_size)
        model.set_n_grams(conditional_probabilities)
        return model

    def get_text_processor(self) -> TextProcessor:  # type: ignore[empty-body]
        """
        Get method for the processor created for the current JSON file.

        Returns:
            TextProcessor: processor created for the current JSON file.
        """
        return self._text_processor

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
        self._text_processor = text_processor
        self._language_models = {model.get_n_gram_size(): model for model in language_models}

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
        if (
            not isinstance(seq_len, int)
            or not isinstance(prompt, str)
            or not prompt
            or not seq_len
            ):
            return None
        encoded_prompt = self._text_processor.encode(prompt)
        if not encoded_prompt:
            return None
        current_sequence = list(encoded_prompt)
        iteration = 1
        while iteration <= seq_len:
            next_token_candidates = self._get_next_token(tuple(current_sequence))
            if next_token_candidates is None or not next_token_candidates:
                break
            max_prob = max(next_token_candidates.values())
            max_token = [token for token, prob in next_token_candidates.items()
                                                        if prob == max_prob][0]
            current_sequence.append(max_token)
            iteration += 1
        return self._text_processor.decode(tuple(current_sequence))

    def _get_next_token(self, sequence_to_continue: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            dict[int, float] | None: Next tokens for sequence continuation

        In case of corrupt input arguments return None.
        """
        if (
            not isinstance(sequence_to_continue, tuple)
            or not sequence_to_continue
            or not self._language_models
            ):
            return None
        n_gram_sizes = sorted(self._language_models.keys(), reverse=True)
        for n_gram_size in n_gram_sizes:
            n_gram_model = self._language_models[n_gram_size]
            token_candidates = n_gram_model.generate_next_token(sequence_to_continue)
            if token_candidates is not None and token_candidates:
                return token_candidates
        return None
