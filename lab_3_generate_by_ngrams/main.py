"""
Lab 3.

Beam-search and natural language generation evaluation
"""

# pylint:disable=too-few-public-methods, unused-import
import json
import math
import string


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
        if (not isinstance(seq_len, int)
            or not isinstance(prompt, str)
            or not prompt):
            return None
        encoded_prompt = self._text_processor.encode(prompt)
        n_gram_size = self._model.get_n_gram_size()
        if n_gram_size is None or encoded_prompt is None:
            return None
        sequence = list(encoded_prompt)
        for _ in range(seq_len):
            context = tuple(sequence[-(n_gram_size - 1):])
            next_token_candidates = self._model.generate_next_token(context)
            if not next_token_candidates:
                break
            next_token = max(next_token_candidates.items(), key=lambda x: x[1])[0]
            sequence.append(next_token)
        decoded_text = self._text_processor.decode(tuple(sequence))
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
        if not isinstance(sequence, tuple) or not sequence:
            return None
        next_token = self._model.generate_next_token(sequence)
        if next_token is None:
            return None
        if not next_token:
            return []
        result = list(next_token.items())
        result.sort(key=lambda x: (-x[1], x[0]))
        return result[:self._beam_width]

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
        if (not isinstance(sequence, tuple)
            or not isinstance(next_tokens, list)
            or not isinstance(sequence_candidates, dict)):
            return None
        if (not next_tokens
            or not sequence_candidates
            or not sequence
            or sequence not in sequence_candidates):
            return None
        if len(next_tokens) > self._beam_width:
            return None
        current_score = sequence_candidates[sequence]
        new_candidates = {seq: score
        for seq, score in sequence_candidates.items() if seq != sequence}
        new_candidates.update(
            (sequence + (token,), current_score - math.log(prob))
            for token, prob in next_tokens)
        return new_candidates

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
        if not isinstance(sequence_candidates, dict):
            return None
        if not sequence_candidates:
            return None
        sorted_sequences = (sorted(sequence_candidates.items(),
        key=lambda x: (x[1], tuple(-element for element in x[0]))))
        result = dict(sorted_sequences[:self._beam_width])
        return result

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
        if not isinstance(seq_len, int) or not isinstance(prompt, str):
            return None
        if not prompt or seq_len <= 0:
            return None
        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None
        sequence_candidates = {encoded_prompt: 0.0}
        for _ in range(seq_len):
            new_candidates = {}
            for sequence, probability in sequence_candidates.items():
                next_tokens = self._get_next_token(sequence)
                if next_tokens is None:
                    return None
                updated_candidates = self.beam_searcher.continue_sequence(
                    sequence, next_tokens, {sequence: probability})
                if updated_candidates is not None:
                    for seq, prob in updated_candidates.items():
                        if seq not in new_candidates or prob < new_candidates[seq]:
                            new_candidates[seq] = prob
            if not new_candidates:
                break
            sequence_candidates = self.beam_searcher.prune_sequence_candidates(new_candidates) or {}
            if sequence_candidates is None:
                return None
        if not sequence_candidates:
            return None
        best_sequence = min(sequence_candidates.items(), key=lambda x: x[1])[0]
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
        if not isinstance(sequence_to_continue, tuple):
            return None
        if not sequence_to_continue:
            return None
        next_token = self.beam_searcher.get_next_token(sequence_to_continue)
        if not next_token:
            return None
        return next_token


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
        with open(json_path, 'r', encoding='utf-8') as file:
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
        if not isinstance(n_gram_size, int):
            return None
        if n_gram_size < 2:
            return None
        n_grams_frequencies = {}
        context_frequencies = {}
        for n_gram, frequency in self._content["freq"].items():
            changed_n_gram = []
            for element in n_gram:
                if element.isalpha() or element == self._eow_token:
                    element_num = self._text_processor.get_id(element.lower())
                    if element_num:
                        changed_n_gram.append(element_num)
                elif element == " ":
                    changed_n_gram.append(0)
            if changed_n_gram:
                n_gram_tuple = tuple(changed_n_gram)
                n_grams_frequencies[n_gram_tuple] = (
                    n_grams_frequencies.get(n_gram_tuple, 0) + frequency)
                if len(changed_n_gram) == n_gram_size:
                    context = n_gram_tuple[:-1]
                    context_frequencies[context] = context_frequencies.get(context, 0) + frequency
        if not n_grams_frequencies:
            return None
        conditional_probabilities = {
            n_gram: n_gram_frequency / context_frequency
            for n_gram, n_gram_frequency in n_grams_frequencies.items()
            for context_frequency in [context_frequencies.get(n_gram[:-1], 0)]
            if context_frequency > 0
        }
        language_model = NGramLanguageModel(None, n_gram_size)
        language_model.set_n_grams(conditional_probabilities)
        return language_model

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
