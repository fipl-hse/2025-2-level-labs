"""
Lab 3.

Beam-search and natural language generation evaluation
"""

# pylint:disable=too-few-public-methods, unused-import
import json
import math

from lab_1_keywords_tfidf.main import check_dict, check_list, check_positive_int


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
            text_without_punctuation = []
            for element in word:
                if element.isalpha():
                    text_without_punctuation.append(element)
            if text_without_punctuation:
                tokens.extend(text_without_punctuation)
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
        if not isinstance (element_id, int):
            return None

        for token, token_id in self._storage.items():
            if token_id == element_id:
                return token

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

        tokenized_text = self._tokenize(text)
        if tokenized_text is None:
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
        if not isinstance(element, str) or len(element) != 1:
            return

        if element not in self._storage:
            self._storage[element] = len(self._storage)


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

        decoded_tokens = self._decode(encoded_corpus)
        if decoded_tokens is None:
            return None

        return self._postprocess_decoded_text(decoded_tokens)

    def fill_from_ngrams(self, content: dict) -> None:
        """
        Fill internal storage with letters from external JSON.

        Args:
            content (dict): ngrams from external JSON
        """

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
        for token in corpus:
            decoded_element = self.get_token(token)
            if decoded_element is None:
                return None
            decoded_corpus.append(decoded_element)

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

        result = []
        spaces = False

        for element in decoded_corpus:
            if element == self._end_of_word_token:
                if not spaces:
                    result.append(' ')
                    spaces = True
            else:
                result.append(element)
                spaces = False

        result_text = ''.join(result).strip()
        if not result_text:
            return None

        result_text = result_text[0].upper() + result_text[1:]

        if result_text and result_text[-1] != '.':
            result_text += '.'

        return result_text


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
        if n_grams is None:
            return 1

        counter_of_n_gram = {}
        counter_of_context = {}

        for n_gram in n_grams:
            counter_of_n_gram[n_gram] = counter_of_n_gram.get(n_gram, 0) + 1
            context = n_gram[:-1]
            counter_of_context[context] = counter_of_context.get(context, 0) + 1

        self._n_gram_frequencies = {}
        for n_gram, value in counter_of_n_gram.items():
            context = n_gram[:-1]
            count_of_context = counter_of_context.get(context, 0)

            if count_of_context > 0:
                frequency = value / count_of_context
                self._n_gram_frequencies[n_gram] = frequency

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
        if not isinstance(sequence, tuple) or not sequence:
            return None

        n_gram_size = self.get_n_gram_size()
        context_length = n_gram_size - 1

        if len(sequence) < n_gram_size - 1:
            return None

        context = tuple(sequence[-(n_gram_size - 1):])
        vocabulary_of_next_token = {}

        for n_gram, frequency in self._n_gram_frequencies.items():
            if n_gram[:context_length] == context:
                next_element = n_gram[-1]
                vocabulary_of_next_token[next_element] = (
                    vocabulary_of_next_token.get(next_element, 0) + frequency
                )

        if vocabulary_of_next_token:
            sorted_tokens = dict(sorted(
                vocabulary_of_next_token.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True
            ))
            return sorted_tokens

        return None

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

        n_size = self._n_gram_size

        if len(encoded_corpus) < n_size:
            return tuple()

        n_grams = []
        for i in range(len(encoded_corpus) - n_size + 1):
            n_gram = encoded_corpus[i:i + n_size]
            n_grams.append(n_gram)

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
        if not isinstance(seq_len, int) or not isinstance(prompt, str):
            return None

        if not prompt.strip():
            return None

        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None

        n_gram_size = self._model.get_n_gram_size()

        sequence = list(encoded_prompt)
        generated_tokens = 0

        while generated_tokens < seq_len:
            if len(sequence) < n_gram_size - 1:
                context = tuple(sequence)
            else:
                start_index = len(sequence) - (n_gram_size - 1)
                context = tuple(sequence[start_index:])

            next_token = self._model.generate_next_token(context)
            if next_token is None:
                break

            if not next_token:
                break

            greedy_generator = max(next_token.items(), key = lambda x: (x[1], x[0]))[0]

            sequence.append(greedy_generator)
            generated_tokens += 1

        decoded_sequence = self._text_processor.decode(tuple(sequence))
        return decoded_sequence


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

        next_tokens = self._model.generate_next_token(sequence)
        if next_tokens is None:
            return None

        if not next_tokens:
            return []

        valid_tokens = []
        for token, probability in next_tokens.items():
            valid_tokens.append((token, probability))

        valid_tokens.sort(key = lambda x: (-x[1], x[0]))
        return valid_tokens[:self._beam_width]

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
        if not (
            isinstance(sequence, tuple)
            and check_list(next_tokens, tuple, False)
            and check_dict(sequence_candidates, tuple, float, True)
            and sequence in sequence_candidates
            and len(next_tokens) <= self._beam_width
        ):
            return None

        current_probability = sequence_candidates[sequence]
        del sequence_candidates[sequence]

        for token, prob in next_tokens:
            if prob <= 0:
                continue
            new_sequence = sequence + (token,)
            frequency = current_probability - math.log(prob)
            sequence_candidates[new_sequence] = frequency

        return sequence_candidates

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
        if not check_dict(sequence_candidates, tuple, float, True) or not sequence_candidates:
            return None

        n_sequences = min(self._beam_width, len(sequence_candidates))
        sorted_sequences = sorted(
            sequence_candidates.items(),
            key = lambda x: x[1],
            reverse = False
            )

        top_n_sequences = sorted_sequences[:n_sequences]

        return dict(top_n_sequences)


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
        if not isinstance(prompt, str) or not prompt.strip():
            return None

        if not check_positive_int(seq_len):
            return None

        encoded_sequence = self._text_processor.encode(prompt)
        if encoded_sequence is None:
            return None

        candidates_of_sequence = {encoded_sequence: 0.0}

        for _ in range(seq_len):
            new_candidates = {}
            continuation_found = False

            for sequence, score in candidates_of_sequence.items():
                next_token_list = self._get_next_token(sequence)
                if next_token_list is None:
                    return None

                if not next_token_list:
                    new_candidates[sequence] = score
                    continue

                updated_candidates = self.beam_searcher.continue_sequence(
                    sequence, next_token_list, {sequence: score}
                )

                if updated_candidates:
                    new_candidates.update(updated_candidates)
                    continuation_found = True

            if not continuation_found:
                break

            prune_candidates = self.beam_searcher.prune_sequence_candidates(new_candidates)
            if prune_candidates is None:
                return None

            candidates_of_sequence = prune_candidates

        if not candidates_of_sequence:
            return None

        best_sequence = min(candidates_of_sequence.items(), key=lambda x: x[1])[0]
        return self._text_processor.decode(best_sequence)

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

    def get_text_processor(self) -> TextProcessor:  # type: ignore[empty-body]
        """
        Get method for the processor created for the current JSON file.

        Returns:
            TextProcessor: processor created for the current JSON file.
        """


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

    def _get_next_token(self, sequence_to_continue: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            dict[int, float] | None: Next tokens for sequence continuation

        In case of corrupt input arguments return None.
        """
