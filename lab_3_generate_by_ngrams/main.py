"""
Lab 3.

Beam-search and natural language generation evaluation
"""

# pylint:disable=too-few-public-methods, unused-import
import json
import math


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
        if not isinstance(text, str):
            return None
        words = text.lower().split()
        tokens = []
        current_word = []
        for word in words:
            current_word = [char for char in word if char.isalpha()]
            if current_word:
                tokens.extend(current_word)
                tokens.append(self._end_of_word_token)
        if not tokens:
            return None
        if (tokens and
            tokens[-1] == self._end_of_word_token and
            (text[-1].isdigit() or text[-1].isalpha())):
            tokens = tokens[:-1]
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
        if (not isinstance(element, str)
            or element not in self._storage):
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
        if not isinstance(text, str):
            return None
        tokens = self._tokenize(text)
        if tokens is None:
            return None
        encoded_tokens = []
        for token in tokens:
            if token not in self._storage:
                self._put(token)
            token_id = self.get_id(token)
            if token_id is None:
                return None
            encoded_tokens.append(token_id)
        return tuple(encoded_tokens)

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): An element to put into storage

        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """
        if not isinstance(element, str) or len(element) != 1 or not element:
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
        if not isinstance(content, dict):
            return
        ngrams_freq = content.get('freq', {})
        for ngram in ngrams_freq.keys():
            for char in ngram:
                if char.isalpha():
                    self._put(char.lower())
                elif char == self._end_of_word_token:
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
        text = ""
        for token in decoded_corpus:
            if token == self._end_of_word_token:
                if not text.endswith(" "):
                    text += " "
            else:
                text += token
        text = text.strip()
        if not text:
            return None
        text = text[0].upper() + text[1:]
        if not text.endswith("."):
            text += "."
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
        if not isinstance(self._encoded_corpus, tuple) or not self._encoded_corpus:
            return 1
        n_gramms = self._extract_n_grams(self._encoded_corpus)
        if n_gramms is None:
            return 1
        n_gram_abs_freqs = {}
        n_gram_prefix_counts = {}
        for n_gram in n_gramms:
            n_gram_abs_freqs[n_gram] = n_gram_abs_freqs.get(n_gram, 0) + 1
            n_gram_prefix_counts[n_gram[:-1]] = n_gram_prefix_counts.get(n_gram[:-1], 0) + 1
        for n_gram, abs_freq in n_gram_abs_freqs.items():
            self._n_gram_frequencies[n_gram] = abs_freq / n_gram_prefix_counts[n_gram[:-1]]
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
            or len(sequence) < self._n_gram_size - 1
        ):
            return None
        context_length = self._n_gram_size - 1
        context = sequence[-context_length:]
        generated_tokens = {}
        for n_gram, probability in self._n_gram_frequencies.items():
            if n_gram[:len(context)] == context:
                if n_gram[-1] not in generated_tokens:
                    generated_tokens[n_gram[-1]] = probability
        if generated_tokens:
            sorted_tokens = dict(sorted(
                generated_tokens.items(),
                key=lambda x: (-x[1], -x[0])
            ))
            return sorted_tokens
        return {}

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
        if not isinstance(seq_len, int) or seq_len <= 0:
            return None
        if not isinstance(prompt, str) or not prompt:
            return None
        encoded_sequence = self._text_processor.encode(prompt)
        if encoded_sequence is None:
            return None
        for _ in range(seq_len):
            candidates = self._model.generate_next_token(encoded_sequence)
            if not candidates:
                break
            next_token = next(iter(candidates.keys()))
            encoded_sequence = encoded_sequence + (next_token,)
        return self._text_processor.decode(encoded_sequence)


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
        next_tokens = list(candidates.items())
        next_tokens.sort(key=lambda x: (-x[1], -x[0]))
        return next_tokens[:self._beam_width]

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
        if not isinstance(sequence, tuple) or not sequence:
            return None
        if not isinstance(next_tokens, list) or not next_tokens:
            return None
        if not isinstance(sequence_candidates, dict):
            return None
        if sequence not in sequence_candidates:
            return None
        if len(next_tokens) > self._beam_width:
            return None
        for token_prob in next_tokens:
            if (not isinstance(token_prob, tuple) or 
                len(token_prob) != 2 or
                not isinstance(token_prob[0], int) or
                not isinstance(token_prob[1], (int, float)) or
                token_prob[1] <= 0):
                return None
        current_prob = sequence_candidates[sequence]
        del sequence_candidates[sequence]
        for token, token_prob in next_tokens:
            new_sequence = sequence + (token,)
            new_prob = current_prob - math.log(token_prob)
            sequence_candidates[new_sequence] = new_prob
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
        if not isinstance(sequence_candidates, dict):
            return None
        if not sequence_candidates:
            return None
        for seq, prob in sequence_candidates.items():
            if not isinstance(seq, tuple) or not isinstance(prob, (int, float)):
                return None
        sorted_candidates = sorted(
            sequence_candidates.items(),
            key=lambda x: (x[1], x[0])
        )
        pruned_candidates = dict(sorted_candidates[:self._beam_width])
        return pruned_candidates


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
        if (not isinstance(prompt, str) 
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
            new_candidates = {}
            has_valid_candidates = False
            for sequence, current_prob in list(sequence_candidates.items()):
                next_tokens = self._get_next_token(sequence)
                if next_tokens is None:
                    return None
                if not next_tokens:
                    new_candidates[sequence] = current_prob
                    has_valid_candidates = True
                    continue
                temp_candidates = {sequence: current_prob}
                result = self.beam_searcher.continue_sequence(
                    sequence, next_tokens, temp_candidates
                )
                if result is None:
                    new_candidates[sequence] = current_prob
                    has_valid_candidates = True
                else:
                    for seq, prob in result.items():
                        new_candidates[seq] = prob
                    has_valid_candidates = True
            if not has_valid_candidates:
                break
            if new_candidates:
                sequence_candidates = self.beam_searcher.prune_sequence_candidates(new_candidates)
                if not sequence_candidates:
                    break
            else:
                break
        if sequence_candidates:
            best_sequence = min(sequence_candidates.items(), key=lambda x: (x[1], x[0]))[0]
            return self._text_processor.decode(best_sequence)
        return None

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
        self._text_processor = TextProcessor(self._eow_token)
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                self._content = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            self._content = {}
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
        ngrams = self._content.get('freq', {})
        ngram_abs_freqs = {}
        ngram_prefix_counts = {}
        for ngram, frequency in ngrams.items():
            encoded_ngram = []
            for symbol in ngram:
                if symbol.isalpha():
                    sym_id = self._text_processor.get_id(symbol.lower())
                elif symbol.isspace():
                    sym_id = self._text_processor.get_id(self._eow_token)
                else:
                    continue
                if sym_id is None:
                    break
                encoded_ngram.append(sym_id)
            if len(encoded_ngram) == n_gram_size:
                ngram_abs_freqs[tuple(encoded_ngram)] = ngram_abs_freqs.get(
                    tuple(encoded_ngram), 0) + frequency
                prefix = tuple(encoded_ngram)[:-1]
                ngram_prefix_counts[prefix] = ngram_prefix_counts.get(prefix, 0) + frequency
        n_gram_frequencies = {}
        for ngram, abs_freq in ngram_abs_freqs.items():
            prefix_count = ngram_prefix_counts.get(ngram[:-1], 0)
            if prefix_count > 0:
                n_gram_frequencies[ngram] = abs_freq / prefix_count
        language_model = NGramLanguageModel(None, n_gram_size)
        language_model.set_n_grams(n_gram_frequencies)
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
        for model in language_models:
            if isinstance(model, NGramLanguageModel):
                self._language_models[model.get_n_gram_size()] = model
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
        if not isinstance(seq_len, int) or seq_len <= 0:
            return None
        if not isinstance(prompt, str) or not prompt:
            return None
        encoded_sequence = self._text_processor.encode(prompt)
        if encoded_sequence is None:
            return None
        for _ in range(seq_len):
            candidates = self._get_next_token(encoded_sequence)
            if candidates is None:
                break
            if not candidates:
                break
            try:
                sorted_candidates = sorted(candidates.items(), key=lambda x: (-x[1], x[0]))
                next_token = sorted_candidates[0][0]
                encoded_sequence = encoded_sequence + (next_token,)
            except (IndexError, TypeError):
                break
        result = self._text_processor.decode(encoded_sequence)
        return result

    def _get_next_token(self, sequence_to_continue: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            dict[int, float] | None: Next tokens for sequence continuation

        In case of corrupt input arguments return None.
        """
        if not isinstance(sequence_to_continue, tuple) or not sequence_to_continue:
            return None
        ngram_sizes = sorted(self._language_models.keys(), reverse=True)
        for ngram_size in ngram_sizes:
            model = self._language_models[ngram_size]
            context_length = ngram_size - 1
            if len(sequence_to_continue) >= context_length:
                context = sequence_to_continue[-context_length:] if context_length > 0 else tuple()
                candidates = model.generate_next_token(context)
                if candidates is not None and candidates:
                    return candidates
        return None
