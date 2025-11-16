"""
Lab 3.

Beam-search and natural language generation evaluation
"""

import json
import math
import string

# pylint:disable=too-few-public-methods, unused-import
from copy import deepcopy

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
            word_tokens = [char for char in word if char.isalpha()]
            tokens.extend(word_tokens)
            if word_tokens:
                tokens.append(self._end_of_word_token)
        if not tokens:
            return None
        if text[-1].isalnum() and tokens[-1] == self._end_of_word_token:
            tokens.pop()
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
        for element, idx in self._storage.items():
            if idx == element_id:
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
        tokens = self._tokenize(text)
        if tokens is None:
            return None
        encoded = []
        for token in tokens:
            self._put(token)
            token_id = self.get_id(token)
            if token_id is None:
                return None
            encoded.append(token_id)
        return tuple(encoded)

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
        decoded = self._decode(encoded_corpus)
        if decoded is None:
            return None
        result = self._postprocess_decoded_text(decoded)
        if result is None:
            return None
        return result

    def fill_from_ngrams(self, content: dict) -> None:
        """
        Fill internal storage with letters from external JSON.

        Args:
            content (dict): ngrams from external JSON
        """
        if not isinstance(content, dict) or not content:
            return

        freq = content.get("freq")

        for ngram in freq.keys():
            for char in ngram.lower():
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
        decoded = []
        for id_ in corpus:
            token = self.get_token(id_)
            if token is None:
                return None
            decoded.append(token)
        return tuple(decoded)

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
        text = ''.join(decoded_corpus).replace('_', ' ').capitalize()
        if text[-1] == ' ':
            text = text[:-1] + '.'
        elif not text.endswith('.'):
            text += '.'
        return text


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
        if not isinstance(self._encoded_corpus, tuple) or len(self._encoded_corpus) == 0:
            return 1
        n_grams = self._extract_n_grams(self._encoded_corpus)
        if n_grams is None:
            return 1
        n_gram_counts = {}
        prefix_counts = {}
        n_gram = self._n_gram_size
        for n_gram in n_grams:
            n_gram_counts[n_gram] = n_gram_counts.get(n_gram, 0) + 1
            prefix = n_gram[:-1]
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        self._n_gram_frequencies = {}
        for n_gram, count in n_gram_counts.items():
            prefix = n_gram[:-1]
            prefix_count = prefix_counts[prefix]
            probability = count / prefix_count
            self._n_gram_frequencies[n_gram] = probability
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
            not isinstance(sequence, tuple) or
            len(sequence) == 0 or
            len(sequence) < self._n_gram_size - 1
        ):
            return None

        context = sequence[-(self._n_gram_size - 1):]

        filtered_tokens = {
            n_gram[-1]: prob
            for n_gram, prob in self._n_gram_frequencies.items()
            if n_gram[:-1] == context
        }

        sorted_items = sorted(
            filtered_tokens.items(),
            key=lambda x: (x[1], x[0]),
            reverse=True
        )

        return dict(sorted_items)

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
        if not isinstance(encoded_corpus, tuple) or len(encoded_corpus) == 0:
            return None
        n = self._n_gram_size
        n_grams = tuple(
            tuple(encoded_corpus[i:i+n])
            for i in range(len(encoded_corpus) - n + 1)
        )
        return n_grams


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
        if not isinstance(seq_len, int) or not isinstance(prompt, str) or len(prompt) == 0:
            return None
        encoded_seq = self._text_processor.encode(prompt)
        if encoded_seq is None or len(encoded_seq) == 0:
            return None
        n = self._model.get_n_gram_size()
        if len(encoded_seq) < n - 1:
            return None
        generated = list(encoded_seq)
        for _ in range(seq_len):
            context = tuple(generated[-(n - 1):])
            candidates = self._model.generate_next_token(context)
            if not candidates:
                break
            next_token = max(candidates.items(), key=lambda x: (x[1], x[0]))[0]
            generated.append(next_token)
        decoded_text = self._text_processor.decode(tuple(generated))
        return decoded_text


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
        if not isinstance(sequence, tuple) or len(sequence) == 0:
            return None
        candidates = self._model.generate_next_token(sequence)
        if candidates is None:
            return None
        if not candidates:
            return []
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: (x[1], x[0]),
            reverse=True
        )
        top_candidates = sorted_candidates[:self._beam_width]
        return top_candidates

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
            not isinstance(sequence, tuple) or
            not isinstance(next_tokens, list) or
            not isinstance(sequence_candidates, dict)
        ):
            return None
        if (            
            len(next_tokens) > self._beam_width or
            sequence not in sequence_candidates or
            len(next_tokens) == 0
        ):
            return None
        updated_candidates = deepcopy(sequence_candidates)
        if sequence in updated_candidates:
            del updated_candidates[sequence]
        base_prob = sequence_candidates[sequence]
        for token, prob in next_tokens:
            if prob <= 0:
                continue
            new_sequence = sequence + (token,)
            new_prob = base_prob - math.log(prob)
            updated_candidates[new_sequence] = new_prob
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
        if not isinstance(sequence_candidates, dict) or len(sequence_candidates) == 0:
            return None
        sorted_candidates = sorted(
            sequence_candidates.items(),
            key=lambda x: (x[1], x[0]),
            reverse=False,
        )
        top_n = sorted_candidates[:self._beam_width]
        return dict(top_n)


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
        if (
                not isinstance(prompt, str)
                or not prompt.strip()
                or not check_positive_int(seq_len)
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

                continued = self.beam_searcher.continue_sequence(sequence,next_tokens, sequence_candidates)
                if continued is None:
                    continue

                pruned_candidates = self.beam_searcher.prune_sequence_candidates(continued)
                if pruned_candidates is None:
                    return None
                sequence_candidates = pruned_candidates

            if not sequence_candidates:
                return None

        best_sequence = min(sequence_candidates.items(), key=lambda x: x[1])[0]
        result = self._text_processor.decode(best_sequence)
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
        if not isinstance(sequence_to_continue, tuple) or len(sequence_to_continue) == 0:
            return None
        next_tokens = self.beam_searcher.get_next_token(sequence_to_continue)
        if next_tokens is None:
            return None
        return next_tokens


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
        if not isinstance(n_gram_size, int) or n_gram_size < 2:
            return None

        if not self._content or "freq" not in self._content:
            return None

        freq_dict = self._content["freq"]
        eow = self._eow_token
        tp = self._text_processor

        ngram_counts = {}
        prefix_counts = {}

        for raw_ngram, count in freq_dict.items():
            cleaned = raw_ngram.replace(" ", eow)

            filtered = "".join(
                ch.lower()
                for ch in cleaned
                if ch.isalpha() or ch == eow
            )

            if len(filtered) != n_gram_size:
                continue

            ngram_ids = []
            for ch in filtered:
                ch_id = tp.get_id(ch)
                if ch_id is None:
                    break
                ngram_ids.append(ch_id)
            else:
                ngram_tuple = tuple(ngram_ids)
                ngram_counts[ngram_tuple] = ngram_counts.get(ngram_tuple, 0) + count

                prefix_tuple = ngram_tuple[:-1]
                prefix_counts[prefix_tuple] = prefix_counts.get(prefix_tuple, 0) + count

        ngram_freqs = {}
        for ngram_tuple, ngram_count in ngram_counts.items():
            prefix_tuple = ngram_tuple[:-1]
            prefix_count = prefix_counts.get(prefix_tuple, 0)
            if prefix_count > 0:
                prob = ngram_count / prefix_count
                ngram_freqs[ngram_tuple] = prob

        model = NGramLanguageModel(encoded_corpus=None, n_gram_size=n_gram_size)
        model.set_n_grams(ngram_freqs)

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
        self._language_models = {model.get_n_gram_size(): model for model in language_models}
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
            or not prompt.strip()
            or seq_len < 0
            ):
            return None
        encoded_sequence = self._text_processor.encode(prompt)
        if not encoded_sequence:
            return None
        encoded_sequence_list = list(encoded_sequence)
        counter = 0
        while counter < seq_len:
            candidates = self._get_next_token(tuple(encoded_sequence_list))
            if not candidates:
                break

            next_token_id = max(candidates, key=lambda k: candidates[k])
            encoded_sequence_list.append(next_token_id)
            counter += 1

        decoded_sequence = self._text_processor.decode(tuple(encoded_sequence_list))
        return decoded_sequence

    def _get_next_token(self, sequence_to_continue: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            dict[int, float] | None: Next tokens for sequence continuation

        In case of corrupt input arguments return None.
        """
        if (not isinstance(sequence_to_continue, tuple) or not
        sequence_to_continue or not self._language_models):
            return None

        ngram_sizes = sorted(self._language_models.keys(), reverse=True)

        for n in ngram_sizes:
            model = self._language_models[n]
            candidates = model.generate_next_token(sequence_to_continue)

            if candidates and candidates is not None:
                return candidates

        return None
