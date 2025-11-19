"""
Lab 3.

Beam-search and natural language generation evaluation
"""

# pylint:disable=too-few-public-methods, unused-import
import json


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

    def get_end_of_word_token(self) -> str:  # type: ignore[empty-body]
        """
        Retrieve value stored in self._end_of_word_token attribute.

        Returns:
            str: EoW token
        """

    def get_token(self, element_id: int) -> str | None:
        """
        Retrieve an element by unique identifier.

        Args:
            element_id (int): Identifier to retrieve identifier for

        Returns:
            str | None: Element that corresponds to the given identifier

        In case of corrupt input arguments or arguments not included in storage, None is returned
        """

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

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): An element to put into storage

        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """

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

    def get_n_gram_size(self) -> int:  # type: ignore[empty-body]
        """
        Retrieve value stored in self._n_gram_size attribute.

        Returns:
            int: Size of stored n_grams
        """

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

    def generate_next_token(self, sequence: tuple[int, ...]) -> dict | None:
        """
        Retrieve tokens that can continue the given sequence along with their probabilities.

        Args:
            sequence (tuple[int, ...]): A sequence to match beginning of NGrams for continuation

        Returns:
            dict | None: Possible next tokens with their probabilities

        In case of corrupt input arguments, None is returned
        """

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
