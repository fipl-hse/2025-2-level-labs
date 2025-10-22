"""
Lab 3.

Beam-search and natural language generation evaluation
"""

# pylint:disable=too-few-public-methods, unused-import
import json
from math import log
from lab_1_keywords_tfidf.main import check_positive_int, check_list, check_dict

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
        if not isinstance(text, str) or not text:
            return None


        tokens = []
        punctuation = "."

        for element in text:
            if element.isalpha():
                tokens.append(element.lower())
            elif element.isspace() or element == punctuation:
                if tokens and tokens[-1] != self._end_of_word_token:
                    tokens.append(self._end_of_word_token)

        if not tokens:
            return None

        return tuple(tokens)

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

        return self._storage[element]



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
        if not (
            isinstance(element_id, int) and not isinstance(element_id, bool)
            and element_id in self._storage.values()
        ):
            return None

        for item, value in self._storage.items():
            if value == element_id:
                return item

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

        for token in tokenized_text:
            self._put(token)

        encoded_text = []
        for token in tokenized_text:
            token_value = self.get_id(token)
            if token_value is None:
                return None
            encoded_text.append(token_value)

        return tuple(encoded_text)

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


        decoded_text = self._decode(encoded_corpus)
        if decoded_text is None:
            return None

        return self._postprocess_decoded_text(decoded_text)

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


        decoded_text = []
        for element in corpus:
            decoded_element = self.get_token(element)
            if decoded_element is None:
                return None
            decoded_text.append(decoded_element)

        return tuple(decoded_text)


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

        decoded_string = ''.join(decoded_corpus).replace(self._end_of_word_token, " ")

        decoded_string = decoded_string.capitalize()

        decoded_string = decoded_string.rstrip()

        decoded_string = decoded_string.capitalize()

        decoded_string += "."

        return decoded_string



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
        self._n_gram_frequencies = None

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

        extracted_n_grams = self._extract_n_grams(self._encoded_corpus)
        if extracted_n_grams is None:
            return 1

        n_gram_counts = {}
        context_counts = {}
        for n_gram in extracted_n_grams:

            if n_gram in n_gram_counts:
                n_gram_counts[n_gram] += 1
            else:
                n_gram_counts[n_gram] = 1

            context = n_gram[:-1]
            if context in context_counts:
                context_counts[context] += 1
            else:
                context_counts[context] = 1

        self._n_gram_frequencies = {
            n_gram: n_gram_counts[n_gram] / context_counts[n_gram[:-1]]
            for n_gram in set(extracted_n_grams)
        }

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
        if not (
            isinstance(sequence, tuple)
            and sequence
            and len(sequence) >= self._n_gram_size - 1
        ):
            return None


        context = sequence[-(self._n_gram_size-1):]

        return {
            n_gram[-1]: frequency
            for n_gram, frequency in self._n_gram_frequencies.items()
            if n_gram[:-1] == context
        }

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

        extracted_n_grams = []
        for index, _ in enumerate(encoded_corpus):
            n_gram = encoded_corpus[index: index + self._n_gram_size]
            if len(n_gram) < self._n_gram_size:
                break
            extracted_n_grams.append(n_gram)

        return tuple(extracted_n_grams)


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
        if not (
            check_positive_int(seq_len)
            and isinstance(prompt, str) and prompt
        ):
            return None

        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None

        generated_sequence = list(encoded_prompt)

        for _ in range(seq_len):

            context = tuple(generated_sequence[-(self._model.get_n_gram_size()-1):])

            candidates = self._model.generate_next_token(context)
            if not candidates:
                break

            token = max(candidates, key=lambda token: (candidates[token], token))

            generated_sequence.append(token)

        return self._text_processor.decode(tuple(generated_sequence))


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


        candidates = self._model.generate_next_token(sequence)
        if candidates is None:
            return None

        if not candidates:
            return []

        best_candidates = sorted(candidates.items(), key=lambda item: item[1], reverse=True)

        return best_candidates[:self._beam_width]


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
            and check_list(next_tokens ,tuple, False)
            and check_dict(sequence_candidates, tuple, float, False)
            and sequence in sequence_candidates
            and len(next_tokens) <= self._beam_width
        ):
            return None

        updated_candidates = {}

        for token, frequency in next_tokens:

            if frequency <= 0:
                return None

            new_sequence = sequence + (token, )
            updated_candidates[new_sequence] = sequence_candidates[sequence] - log(frequency)

        return updated_candidates


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
        if not check_dict(sequence_candidates, tuple, float, False):
            return None


        sequence_candidates = sorted(
            sequence_candidates.items(),
            key=lambda item: item[1]
        )

        return dict(sequence_candidates[:self._beam_width])


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
        self._language_model = language_model
        self._text_processor = text_processor
        self._beam_width = beam_width
        self.beam_searcher = BeamSearcher(beam_width, language_model)


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
        if not (
            check_positive_int(seq_len)
            and isinstance(prompt, str) and prompt
        ):
            return None


        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None

        candidates = {encoded_prompt: 0.0}

        for _ in range(seq_len):

            extended_candidates = {}

            for sequence in tuple(candidates.keys()):

                next_tokens = self._get_next_token(sequence)
                if next_tokens is None:
                    return None

                new_candidates = self.beam_searcher.continue_sequence(
                    sequence, next_tokens, candidates
                )

                if new_candidates is not None:
                    extended_candidates.update(new_candidates)

            if not extended_candidates:
                break

            candidates = self.beam_searcher.prune_sequence_candidates(extended_candidates)
            if candidates is None:
                return None


        return self._text_processor.decode(min(candidates, key=candidates.get))


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


        return self.beam_searcher.get_next_token(sequence_to_continue) or None


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
