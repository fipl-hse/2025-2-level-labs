"""
Lab 3.

Beam-search and natural language generation evaluation
"""

# pylint:disable=too-few-public-methods, unused-import
import json
import math

from lab_1_keywords_tfidf.main import (
    check_dict,
    check_list,
    check_positive_int
)


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

        Punctuation and digits are removed. EoW token is appended after
        the last word in two cases:
        1. It is followed by punctuation
        2. It is followed by space symbol

        Args:
            text (str): Original text

        Returns:
            tuple[str, ...] | None: Tokenized text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not text or not isinstance(text, str):
            return None

        processed_text = text.lower()
        tokens = []
        word_started = False

        for character in processed_text:
            if character.isalpha():
                tokens.append(character)
                word_started = True
            elif word_started and character.isspace():
                tokens.append(self._end_of_word_token)
                word_started = False

        if word_started:
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
            int | None: Integer identifier that corresponds
            to the given element

        In case of corrupt input arguments or
        arguments not included in storage,
        None is returned
        """
        if not isinstance(element, str):
            return None

        if element in self._storage:
            identifier = self._storage[element]
            return identifier

        return None

    def get_end_of_word_token(self) -> str:
        """
        Retrieve value stored in self._end_of_word_token attribute.

        Returns:
            str: EoW token
        """
        return str(self._end_of_word_token)

    def get_token(self, element_id: int) -> str | None:
        """
        Retrieve an element by unique identifier.

        Args:
            element_id (int): Identifier to retrieve identifier for

        Returns:
            str | None: Element that corresponds to the given identifier

        In case of corrupt input arguments or
        arguments not included in storage, None is returned
        """
        if (
            not isinstance(element_id, int)
            or element_id >= len(self._storage)
        ):
            return None
        for token, token_id in self._storage.items():
            if token_id == element_id:
                return str(token)
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
        if not text or not isinstance(text, str):
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
        if (
            isinstance(element, str)
            and len(element) == 1
            and element not in self._storage
        ):
            self._storage[element] = len(self._storage)

    def decode(self, encoded_corpus: tuple[int, ...]) -> str | None:
        """
        Decode and postprocess encoded corpus by
        converting integer identifiers to string.

        Special symbols are replaced with spaces
        (no multiple spaces in a row are allowed).
        The first letter is capitalized, resulting sequence
        must end with a full stop.

        Args:
            encoded_corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            str | None: Resulting text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not encoded_corpus or not isinstance(encoded_corpus, tuple):
            return None
        decoded_text = self._decode(encoded_corpus)
        if decoded_text is None:
            return None
        resulting_text = self._postprocess_decoded_text(decoded_text)
        if resulting_text is None:
            return None
        return resulting_text

    def fill_from_ngrams(self, content: dict) -> None:
        """
        Fill internal storage with letters from external JSON.

        Args:
            content (dict): ngrams from external JSON
        """
        if isinstance(content, dict) and content:
            ngrams_freq = content.get("freq")
            if check_dict(ngrams_freq, str, int, False) and ngrams_freq:
                for ngram in ngrams_freq.keys():
                    for char in ngram.lower():
                        if char.isalpha() or char == self._end_of_word_token:
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
        if not corpus or not isinstance(corpus, tuple):
            return None
        decoded_corpus = []
        for code in corpus:
            decoded_token = self.get_token(code)
            if decoded_token is None:
                return None
            decoded_corpus.append(decoded_token)
        return tuple(decoded_corpus)

    def _postprocess_decoded_text(
            self, decoded_corpus: tuple[str, ...]
    ) -> str | None:
        """
        Convert decoded sentence into the string sequence.

        Special symbols are replaced with spaces
        (no multiple spaces in a row are allowed).
        The first letter is capitalized,
        resulting sequence must end with a full stop.

        Args:
            decoded_corpus (tuple[str, ...]): A tuple of decoded tokens

        Returns:
            str | None: Resulting text

        In case of corrupt input arguments, None is returned
        """
        if not decoded_corpus or not isinstance(decoded_corpus, tuple):
            return None
        postprocessed_text = ''.join(decoded_corpus)
        postprocessed_text = postprocessed_text.replace('_', ' ')
        postprocessed_text = postprocessed_text[:-1].capitalize() + '.'
        return postprocessed_text


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

    def get_n_gram_size(self) -> int:
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
        if check_dict(frequencies, tuple, float, False) and frequencies:
            self._n_gram_frequencies = frequencies

    def build(self) -> int:
        """
        Fill attribute `_n_gram_frequencies` from encoded corpus.

        Encoded corpus is stored in the attribute `_encoded_corpus`

        Returns:
            int: 0 if attribute is filled successfully, otherwise 1

        In case of corrupt input arguments or methods used return None,
        1 is returned
        """
        if not isinstance(
            self._encoded_corpus, tuple
        ) or not self._encoded_corpus:
            return 1

        extracted_ngrams = self._extract_n_grams(self._encoded_corpus)
        if extracted_ngrams is None:
            return 1

        frequency_counts = {}
        prefix_counts = {}

        for ngram in extracted_ngrams:
            frequency_counts[ngram] = frequency_counts.get(ngram, 0) + 1
            prefix = ngram[:-1]
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

        for ngram, count in frequency_counts.items():
            prefix = ngram[:-1]
            prefix_count = prefix_counts.get(prefix, 0)
            if prefix_count > 0:
                probability_value = count / prefix_count
                self._n_gram_frequencies[ngram] = probability_value

        return 0

    def generate_next_token(self, sequence: tuple[int, ...]) -> dict | None:
        """
        Retrieve tokens that can continue the given sequence
        along with their probabilities.

        Args:
            sequence (tuple[int, ...]): A sequence to match
            beginning of NGrams for continuation

        Returns:
            dict | None: Possible next tokens with their probabilities

        In case of corrupt input arguments, None is returned
        """
        if not isinstance(sequence, tuple):
            return None
        if not sequence:
            return None
        if len(sequence) < self._n_gram_size - 1:
            return None

        context_length = self._n_gram_size - 1
        context_part = sequence[-context_length:]
        possible_tokens = {}

        for ngram, prob in self._n_gram_frequencies.items():
            ngram_prefix = ngram[:context_length]
            if ngram_prefix == context_part:
                last_token = ngram[-1]
                possible_tokens[last_token] = prob

        return possible_tokens

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
        n_grams = []
        for code_index in range(len(encoded_corpus) - self._n_gram_size + 1):
            n_gram = encoded_corpus[code_index:code_index + self._n_gram_size]
            n_grams.append(n_gram)
        return tuple(n_grams)


class GreedyTextGenerator:
    """
    Greedy text generation by N-grams.

    Attributes:
        _model (NGramLanguageModel): A language model
        to use for text generation
        _text_processor (TextProcessor): A TextProcessor instance
        to handle text processing
    """

    def __init__(
            self, language_model: NGramLanguageModel,
            text_processor: TextProcessor
    ) -> None:
        """
        Initialize an instance of GreedyTextGenerator.

        Args:
            language_model (NGramLanguageModel): A language model
            to use for text generation
            text_processor (TextProcessor): A TextProcessor instance
            to handle text processing
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
            not isinstance(prompt, str)
            or not prompt
            or not check_positive_int(seq_len)
        ):
            return None
        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None
        generated_seq = list(encoded_prompt)
        for _ in range(seq_len):
            candidates = self._model.generate_next_token(tuple(generated_seq))
            if not candidates:
                break
            best_candidate = sorted(
                candidates.items(),
                key=lambda x: (x[1], x[0]),
                reverse=True
            )[0][0]
            generated_seq.append(best_candidate)
        decoded_seq = self._text_processor.decode(tuple(generated_seq))
        return decoded_seq


class BeamSearcher:
    """
    Beam Search algorithm for diverse text generation.

    Attributes:
        _beam_width (int): Number of candidates to consider at each step
        _model (NGramLanguageModel): A language model
        to use for next token prediction
    """

    def __init__(
            self, beam_width: int, language_model: NGramLanguageModel
    ) -> None:
        """
        Initialize an instance of BeamSearchAlgorithm.

        Args:
            beam_width (int): Number of candidates to consider at each step
            language_model (NGramLanguageModel): A language model
            to use for next token prediction
        """
        self._beam_width = beam_width
        self._model = language_model

    def get_next_token(
            self, sequence: tuple[int, ...]
    ) -> list[tuple[int, float]] | None:
        """
        Retrieve candidate tokens for sequence continuation.

        The valid candidate tokens are those that are
        included in the N-gram with.
        Number of tokens retrieved must not be bigger
        that beam width parameter.

        The return value has the following format: [(token, probability), ...].
        The return value length matches the Beam Size parameter.

        Args:
            sequence (tuple[int, ...]): Base sequence to continue

        Returns:
            list[tuple[int, float]] | None: Tokens to use
            for base sequence continuation

        In case of corrupt input arguments or methods used return None.
        """
        if (
            not isinstance(sequence, tuple)
            or not sequence
        ):
            return None
        candidates_for_generation = self._model.generate_next_token(sequence)
        if candidates_for_generation is None:
            return None
        if not candidates_for_generation:
            return []
        return sorted(list(candidates_for_generation.items()),
                      key=lambda x: x[1], reverse=True)[:self._beam_width]

    def continue_sequence(
        self,
        sequence: tuple[int, ...],
        next_tokens: list[tuple[int, float]],
        sequence_candidates: dict[tuple[int, ...], float],
    ) -> dict[tuple[int, ...], float] | None:
        """
        Generate new sequences from the base sequence
        with next tokens provided.

        The base sequence is deleted after continued variations are added.

        Args:
            sequence (tuple[int, ...]): Base sequence to continue
            next_tokens (list[tuple[int, float]]): Token for
            sequence continuation
            sequence_candidates (dict[tuple[int, ...], float]):
                Storage with all sequences generated

        Returns:
            dict[tuple[int, ...], float] | None: Updated sequence candidates

        In case of corrupt input arguments or unexpected behaviour
        of methods used return None.
        """
        if (
            not isinstance(sequence, tuple)
            or not sequence
            or not check_list(next_tokens, tuple, False)
            or not next_tokens
        ):
            return None
        if (
            not check_dict(sequence_candidates, tuple, float, False)
            or sequence not in sequence_candidates
            or len(next_tokens) > self._beam_width
        ):
            return None
        seq_probability = sequence_candidates[sequence]
        if not isinstance(seq_probability, (int, float)):
            return None
        for token in next_tokens:
            updated_seq = sequence + (token[0],)
            updated_probability = sequence_candidates[sequence] - math.log(
                token[1])
            sequence_candidates[updated_seq] = updated_probability
        del sequence_candidates[sequence]
        return sequence_candidates

    def prune_sequence_candidates(
        self, sequence_candidates: dict[tuple[int, ...], float]
    ) -> dict[tuple[int, ...], float] | None:
        """
        Remove those sequence candidates that do not make
        top-N most probable sequences.

        Args:
            sequence_candidates (dict[tuple[int, ...], float]):
            Current candidate sequences

        Returns:
            dict[tuple[int, ...], float] | None: Pruned sequences

        In case of corrupt input arguments return None.
        """
        if (
            not check_dict(sequence_candidates, tuple, float, False)
            or not sequence_candidates
        ):
            return None
        sorted_candidates = sorted(
            sequence_candidates.items(), key=lambda x: x[1])
        return dict(sorted_candidates[:self._beam_width])


class BeamSearchTextGenerator:
    """
    Class for text generation with BeamSearch.

    Attributes:
        _language_model (tuple[NGramLanguageModel]):
        Language models for next token prediction
        _text_processor (NGramLanguageModel): A TextProcessor instance
        to handle text processing
        _beam_width (NGramLanguageModel):
        Beam width parameter for generation
        beam_searcher (NGramLanguageModel): Searcher instances
        for each language model
    """

    def __init__(
        self, language_model: NGramLanguageModel,
        text_processor: TextProcessor,
        beam_width: int
    ) -> None:
        """
        Initializes an instance of BeamSearchTextGenerator.

        Args:
            language_model (NGramLanguageModel): Language model
            to use for text generation
            text_processor (TextProcessor): A TextProcessor instance
            to handle text processing
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
            or not prompt
            or not isinstance(seq_len, int)
            or seq_len <= 0
        ):
            return None
        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None
        sequence_candidates = {encoded_prompt: 0.0}
        for _ in range(seq_len):
            next_sequence_candidates = {}
            for sequence, probability in sequence_candidates.items():
                next_tokens = self._get_next_token(sequence)
                if not next_tokens:
                    return None
                updated_candidates = self.beam_searcher.continue_sequence(
                    sequence, next_tokens,
                    {sequence: probability}
                )
                if updated_candidates:
                    next_sequence_candidates.update(updated_candidates)
            if not next_sequence_candidates:
                break
            pruned_candidates = self.beam_searcher.prune_sequence_candidates(
                next_sequence_candidates)
            if pruned_candidates is None:
                return None
            sequence_candidates = pruned_candidates
            if not sequence_candidates:
                return None
        best_sequence = min(sequence_candidates.items(), key=lambda x: x[1])[0]
        decoded_sequence = self._text_processor.decode(best_sequence)
        return decoded_sequence

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
        if not isinstance(sequence_to_continue, tuple):
            return None
        if not sequence_to_continue:
            return None

        tokens_data = self.beam_searcher.get_next_token(sequence_to_continue)
        return tokens_data


class NGramLanguageModelReader:
    """
    Factory for loading language models ngrams from external JSON.

    Attributes:
        _json_path (str): Local path to assets file
        _eow_token (str): Special token for text processor
        _text_processor (TextProcessor): A TextProcessor instance
        to handle text processing
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
        self._text_processor = TextProcessor(self._eow_token)
        with open(json_path, 'r', encoding='utf-8') as file:
            self._content = json.load(file)
        self._text_processor.fill_from_ngrams(self._content)

    def load(self, n_gram_size: int) -> NGramLanguageModel | None:
        """
        Fill attribute `_n_gram_frequencies` from dictionary with N-grams.

        The N-grams taken from dictionary must be cleaned
        from digits and punctuation,
        their length must match n_gram_size, and spaces
        must be replaced with EoW token.

        Args:
            n_gram_size (int): Size of ngram

        Returns:
            NGramLanguageModel | None: Built language model.

        In case of corrupt input arguments or unexpected behaviour
        of methods used, return 1.
        """
        if not isinstance(n_gram_size, int):
            return None
        if n_gram_size < 2:
            return None

        ngrams_data = self._content.get('freq', {})
        absolute_frequencies = {}
        prefix_frequencies = {}

        for ngram_text, freq_value in ngrams_data.items():
            encoded_ngram = []

            for char in ngram_text:
                if char.isalpha():
                    char_id = self._text_processor.get_id(char.lower())
                elif char.isspace():
                    char_id = self._text_processor.get_id(self._eow_token)
                else:
                    continue

                if char_id is None:
                    break
                encoded_ngram.append(char_id)

            if len(encoded_ngram) == n_gram_size:
                ngram_tuple = tuple(encoded_ngram)
                current_freq = absolute_frequencies.get(ngram_tuple, 0)
                absolute_frequencies[ngram_tuple] = current_freq + freq_value

                prefix_tuple = ngram_tuple[:-1]
                current_prefix_freq = prefix_frequencies.get(prefix_tuple, 0)
                prefix_frequencies[prefix_tuple] = (
                    current_prefix_freq + freq_value
                )

        final_frequencies = {}
        for ngram_tuple, abs_freq in absolute_frequencies.items():
            prefix_key = ngram_tuple[:-1]
            prefix_freq = prefix_frequencies.get(prefix_key, 0)

            if prefix_freq > 0:
                probability = abs_freq / prefix_freq
                final_frequencies[ngram_tuple] = probability

        model_instance = NGramLanguageModel(None, n_gram_size)
        model_instance.set_n_grams(final_frequencies)
        return model_instance

    def get_text_processor(self) -> TextProcessor:
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
        _language_models (dict[int, NGramLanguageModel]):
        Language models for next token prediction
        _text_processor (NGramLanguageModel): A TextProcessor instance
        to handle text processing
    """

    def __init__(
        self,
        language_models: tuple[NGramLanguageModel, ...],
        text_processor: TextProcessor
    ) -> None:
        """
        Initializes an instance of BackOffGenerator.

        Args:
            language_models (tuple[NGramLanguageModel, ...]):
                Language models to use for text generation
            text_processor (TextProcessor): A TextProcessor instance
            to handle text processing
        """
        self._language_models = {
            model.get_n_gram_size(): model for model in language_models}
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
            or seq_len <= 0
            or not isinstance(prompt, str)
            or not prompt
        ):
            return None
        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None
        generated_seq = list(encoded_prompt)
        for _ in range(seq_len):
            candidates = self._get_next_token(tuple(generated_seq))
            if candidates is None or not candidates:
                break
            sorted_candidates = sorted(
                candidates.items(), key=lambda x: (-x[1], -x[0]))
            best_candidate = sorted_candidates[0][0]
            generated_seq.append(best_candidate)
        return self._text_processor.decode(tuple(generated_seq))

    def _get_next_token(
        self,
        sequence_to_continue: tuple[int, ...]
    ) -> dict[int, float] | None:
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
        if len(sequence_to_continue) == 0:
            return None

        available_sizes = list(self._language_models.keys())
        available_sizes.sort(reverse=True)

        for model_size in available_sizes:
            required_length = model_size - 1
            if len(sequence_to_continue) < required_length:
                continue

            current_model = self._language_models[model_size]
            possible_tokens = current_model.generate_next_token(
                sequence_to_continue)

            if possible_tokens is None:
                continue

            if len(possible_tokens) > 0:
                return possible_tokens

        return None
