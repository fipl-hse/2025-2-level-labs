"""
Lab 3.

Beam-search and natural language generation evaluation
"""

# pylint:disable=too-few-public-methods, unused-import
import json
from math import log

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

        tokens_filtered = []
        for word in text.lower().split():
            tokens = [i for i in word if i.isalpha()]
            tokens_filtered.extend(tokens)
            if tokens:
                tokens_filtered.append(self._end_of_word_token)
        if not tokens_filtered:
            return None
        if text and text[-1].isalpha():
            tokens_filtered.pop()
        return tuple(tokens_filtered)

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
        if not isinstance(element_id, int):
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
        encoded_text = []
        if not tokenized_text:
            return None
        for token in tokenized_text:
            self._put(token)
            token_id = self.get_id(token)
            if token_id is None:
                return None
            encoded_text.append(token_id)
        return tuple(encoded_text)

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.
        Args:
            element (str): An element to put into storage
        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """
        if not isinstance(element, str) or len(element) > 1 or element in self._storage:
            return None
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
        if not isinstance(encoded_corpus, tuple) and encoded_corpus is not None:
            return None
        encoded_letters = self._decode(encoded_corpus)
        if not encoded_letters:
            return None

        encoded_text = self._postprocess_decoded_text(encoded_letters)
        return encoded_text
    def fill_from_ngrams(self, content: dict) -> None:
        """
        Fill internal storage with letters from external JSON.
        Args:
            content (dict): ngrams from external JSON
        """
        if not isinstance(content, dict) or not content:
            return
        for ngram in content['freq']:
            if isinstance(ngram, str):
                ngram = ngram.lower()
                for token in ngram:
                    if token.isalpha():
                        self._put(token)

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
        if not isinstance(corpus, tuple) or corpus == ():
            return None
        decoded_tokens_letters = []
        if corpus is None:
            return None
        for i in corpus:
            if i is None:
                return None
            if self.get_token(i) is None:
                return None
            decoded_tokens_letters.append(self.get_token(i))
        decoded_tokens = tuple(decoded_tokens_letters)
        return decoded_tokens

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
        decoded_text = ""
        for j, i in enumerate(decoded_corpus):
            if j == 0:
                decoded_text += i.upper()
                continue
            if i == self._end_of_word_token:
                decoded_text += " "
                continue
            decoded_text += i
        decoded_text = decoded_text.strip()
        decoded_text += "."
        return decoded_text

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
            return

        self._n_gram_frequencies = frequencies

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
            context = n_gram[:-1]
            n_gram_counts[n_gram] = n_gram_counts.get(n_gram, 0) + 1
            context_counts[context] = context_counts.get(context, 0) + 1

        self._n_gram_frequencies = {
            n_gram: count / context_counts[n_gram[:-1]]
            for n_gram, count in n_gram_counts.items()
            if context_counts.get(n_gram[:-1], 0)
        }

        if not self._n_gram_frequencies:
            return 1

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
            isinstance(sequence, tuple) and sequence and len(sequence) >= self._n_gram_size - 1
        ):
            return None
        context = sequence[-(self._n_gram_size - 1) :]
        next_token = {
            ngram[-1]: frequency
            for ngram, frequency in self._n_gram_frequencies.items()
            if ngram[:-1] == context
        }
        return next_token or None

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
        len_of_corpus = len(encoded_corpus) - self._n_gram_size + 1
        if len_of_corpus <= 0:
            return None
        extracted_n_grams = [
            encoded_corpus[i : i + self._n_gram_size] for i in range(len_of_corpus)
        ]
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
        if not (check_positive_int(seq_len) and isinstance(prompt, str) and prompt):
            return None
        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None
        generated_sequence = list(encoded_prompt)

        for _ in range(seq_len):
            context = tuple(generated_sequence[-(self._model.get_n_gram_size() - 1) :])
            candidates = self._model.generate_next_token(context)
            if not candidates:
                break
            token, _ = max(candidates.items(), key=lambda item: (item[1], item[0]))
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
        return best_candidates[: self._beam_width]

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
            and check_dict(sequence_candidates, tuple, float, False)
            and sequence in sequence_candidates
            and len(next_tokens) <= self._beam_width
        ):
            return None
        updated = sequence_candidates.copy()
        for token, frequency in next_tokens:
            if frequency == 0:
                continue
            new_sequence = sequence + (token,)
            updated[new_sequence] = sequence_candidates[sequence] - log(frequency)
        del updated[sequence]
        return updated

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
        sorted_candidates = sorted(sequence_candidates.items(), key=lambda item: item[1])
        pruned = sorted_candidates[: self._beam_width]
        return dict(pruned)


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
        if not (check_positive_int(seq_len) and isinstance(prompt, str) and prompt):
            return None
        encoded_prompt = self._text_processor.encode(prompt)
        if not encoded_prompt:
            return None
        candidates = {encoded_prompt: 0.0}
        for _ in range(seq_len):
            for sequence in candidates:
                next_tokens = self._get_next_token(sequence)
                if next_tokens is None:
                    return None
                new_candidates = self.beam_searcher.continue_sequence(
                    sequence, next_tokens, candidates
                )
                if new_candidates is None:
                    continue
                pruned = self.beam_searcher.prune_sequence_candidates(new_candidates)
                if pruned is None:
                    return None
                candidates = pruned
            if not candidates:
                return None
        return self._text_processor.decode(min(candidates.items(), key=lambda item: item[1])[0])

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
        self._text_processor = TextProcessor(eow_token)
        with open(self._json_path, "r", encoding="utf-8") as file:
            self._content = json.load(file)
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
        if not check_positive_int(n_gram_size) or n_gram_size < 2:
            return None
        ngrams_freqs = {}
        context_freqs = {}
        for ngram, frequency in self._content["freq"].items():
            processed_ngram = []
            for letter in ngram:
                if letter.isspace():
                    processed_ngram.append(0)
                elif letter.isalpha():
                    symbol_id = self._text_processor.get_id(letter.lower())
                    if symbol_id:
                        processed_ngram.append(symbol_id)
            if processed_ngram:
                ngram_tuple = tuple(processed_ngram)
                ngrams_freqs[ngram_tuple] = (
                    ngrams_freqs.get(ngram_tuple, 0.0) + frequency
                )
                if len(processed_ngram) == n_gram_size:
                    context = ngram_tuple[:-1]
                    context_freqs[context] = context_freqs.get(context, 0.0) + frequency
        if not ngrams_freqs:
            return None
        probs = {}
        for ngram, ngram_freq in ngrams_freqs.items():
            context_freq = context_freqs.get(ngram[:-1], 0.0)
            if context_freq > 0:
                probs[ngram] = ngram_freq / context_freq
        model = NGramLanguageModel(None, n_gram_size)
        model.set_n_grams(probs)
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
        self._language_models = {
            language_model.get_n_gram_size(): language_model for language_model in language_models
        }
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
        if not (check_positive_int(seq_len) and isinstance(prompt, str) and prompt):
            return None

        encoded = self._text_processor.encode(prompt)
        if encoded is None:
            return None

        generated_sequence = list(encoded)

        for _ in range(seq_len):

            candidates = self._get_next_token(tuple(generated_sequence))
            if not candidates:
                break

            token, _ = max(candidates.items(), key=lambda item: (item[1], item[0]))

            generated_sequence.append(token)

        return self._text_processor.decode(tuple(generated_sequence))

    def _get_next_token(self, sequence_to_continue: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve next tokens for sequence continuation.
        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue
        Returns:
            dict[int, float] | None: Next tokens for sequence continuation
        In case of corrupt input arguments return None.
        """
        if not (
            isinstance(sequence_to_continue, tuple)
            and sequence_to_continue
            and self._language_models
        ):
            return None
        sizes = sorted(self._language_models.keys(), reverse=True)
        for i in sizes:
            language_model = self._language_models[i]
            candidates = language_model.generate_next_token(sequence_to_continue)
            if candidates is not None and candidates:
                return candidates
        return None
    