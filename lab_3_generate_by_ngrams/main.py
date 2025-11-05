"""
Lab 3.

Beam-search and natural language generation evaluation
"""

# pylint:disable=too-few-public-methods, unused-import
import json

from lab_1_keywords_tfidf.main import check_dict, check_list

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
        text = text.lower()
        cleaned = ""
        for ch in text:
            if ch.isalpha() or ch.isspace():
                cleaned += ch
            else:
                cleaned += " "
        words = cleaned.split()
        if not words:
            return None
        tokens = []
        for _, word in enumerate(words):
            for letter in word:
                if letter.isalpha():
                    tokens.append(letter)
            tokens.append(self._end_of_word_token)
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
        if not isinstance(element_id, int):
            return None
        for key, value in self._storage.items():
            if value == element_id:
                return key
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
        return self._storage[element] 

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
        if not check_dict(content, tuple, int, False):
            return None
        for ngram in content:
            for letter in ngram:
                if isinstance(letter, str) and len(letter) == 1 and letter.isalpha():
                    self._put(letter)

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
        for idx in corpus:
            token = self.get_token(idx)
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
        result = ''
        for token in decoded_corpus:
            if token == self._end_of_word_token:
                if not result.endswith(' '):
                    result += ' '
            else:
                result += token
        result = result.strip()
        if not result:
            return None
        result = result[0].upper() + result[1:]
        if not result.endswith('.'):
            result += '.'
        return result


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
        if not isinstance(encoded_corpus, tuple) or not isinstance(n_gram_size, int) or n_gram_size <= 0:
            self._encoded_corpus = None
            self._n_gram_size = 0
            self._n_gram_frequencies = {}
            return
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
        if check_dict(frequencies, tuple, (int, float), False):
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
        ngrams = self._extract_n_grams(self._encoded_corpus)
        if ngrams is None:
            return 1
        ngram_counts = {}
        for ngram in ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        context_counts = {}
        for ngram, count in ngram_counts.items():
            context = ngram[:-1]
            context_counts[context] = context_counts.get(context, 0) + count
        self._n_gram_frequencies = {}
        for ngram, count in ngram_counts.items():
            context = ngram[:-1]
            context_count = context_counts.get(context, 0)
            if context_count > 0:
                probability = count / context_count
                self._n_gram_frequencies[ngram] = probability
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
        if not isinstance(sequence, tuple):
            return None
        context_length = self._n_gram_size - 1
        if len(sequence) >= context_length:
            context = sequence[-context_length:]
        else:
            context = sequence
        candidates = {}
        for ngram, prob in self._n_gram_frequencies.items():
            if len(ngram) == self._n_gram_size and ngram[:-1] == context:
                next_token = ngram[-1]
                candidates[next_token] = prob
        return candidates if candidates else None

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
        if (not isinstance(encoded_corpus, tuple) or 
            not isinstance(self._n_gram_size, int) or 
            self._n_gram_size <= 0 or
            len(encoded_corpus) < self._n_gram_size):
            return None
        ngrams = []
        for i in range(len(encoded_corpus) - self._n_gram_size + 1):
            ngram = encoded_corpus[i:i + self._n_gram_size]
            ngrams.append(ngram)
        return tuple(ngrams)


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
        if not isinstance(seq_len, int) or seq_len <= 0 or not isinstance(prompt, str):
            return None
        encoded = self._text_processor.encode(prompt)
        if encoded is None:
            return None
        result = list(encoded)
        for _ in range(seq_len):
            probs = self._model.generate_next_token(tuple(result))
            if not probs:
                break
            sorted_tokens = sorted(probs.items(), key=lambda x: (-x[1], -x[0]))
            next_token = sorted_tokens[0][0]
            result.append(next_token)
            end_of_word_id = self._text_processor.get_id(self._text_processor.get_end_of_word_token())
            if next_token == end_of_word_id:
                break
        decoded = self._text_processor.decode(tuple(result))
        return decoded


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
        if isinstance(beam_width, int) and beam_width > 0:
            self._beam_width = beam_width
        else:
            self._beam_width = 0
        if isinstance(language_model, NGramLanguageModel):
            self._model = language_model
        else:
            self._model = None

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
        if not isinstance(sequence, tuple) or not sequence or self._model is None:
            return None
        probs = self._model.generate_next_token(sequence)
        if probs is None:
            return None
        if check_dict(probs, int, float, True):
            sorted_tokens = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            beam_size = self._beam_width if self._beam_width > 0 else len(sorted_tokens)
            top_tokens = sorted_tokens[:beam_size]
            return top_tokens
        if check_list(probs, tuple, True):
            return probs
        return []

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
        if not check_list(next_tokens, tuple, False):
            return None
        if len(next_tokens) > self._beam_width:
            return None
        for item in next_tokens:
            if not isinstance(item, tuple) or len(item) != 2:
                return None
            if not isinstance(item[0], int) or not isinstance(item[1], float):
                return None
        if not check_dict(sequence_candidates, tuple, float, True):
            return None
        if sequence not in sequence_candidates:
            return None
        base_prob = sequence_candidates[sequence]
        new_candidates = sequence_candidates.copy()
        del new_candidates[sequence]
        for token, prob in next_tokens:
            new_seq = sequence + (token,)
            import math
            if prob > 0:
                new_prob = base_prob - math.log(prob)
            else:
                new_prob = base_prob - math.log(1e-10)  # избегаем log(0)
            new_candidates[new_seq] = new_prob
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
        if not check_dict(sequence_candidates, tuple, float, False):
            return None
        sorted_candidates = sorted(
            sequence_candidates.items(), 
            key=lambda x: (x[1], -sum(x[0]))
        )
        if self._beam_width > 0:
            sorted_candidates = sorted_candidates[:self._beam_width]
        return dict(sorted_candidates)


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
        if not isinstance(language_model, NGramLanguageModel):
            self._language_model = None
            self._text_processor = None
            self._beam_width = 0
            self.beam_searcher = None
            return
        if not isinstance(text_processor, TextProcessor):
            self._language_model = None
            self._text_processor = None
            self._beam_width = 0
            self.beam_searcher = None
            return
        if not isinstance(beam_width, int) or beam_width <= 0:
            beam_width = 1
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
        if not isinstance(prompt, str) or not prompt:
            return None
        if not isinstance(seq_len, int) or seq_len <= 0:
            return None
        if not isinstance(self._text_processor, TextProcessor):
            return None
        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None
        sequence_candidates = {encoded_prompt: 0.0}
        for _ in range(seq_len):
            new_candidates = {}
            for seq in list(sequence_candidates.keys()):
                next_tokens = self._get_next_token(seq)
                if next_tokens is None:
                    continue
                if not next_tokens:
                    continue
                updated = self.beam_searcher.continue_sequence(seq, next_tokens, sequence_candidates)
                if updated is not None:
                    for new_seq, prob in updated.items():
                        if new_seq not in new_candidates or prob < new_candidates[new_seq]:
                            new_candidates[new_seq] = prob
            if not new_candidates:
                break
            pruned = self.beam_searcher.prune_sequence_candidates(new_candidates)
            if pruned is None:
                break
            sequence_candidates = pruned
        if not sequence_candidates:
            return None
        best_sequence = min(sequence_candidates, key=sequence_candidates.get)
        decoded = self._text_processor.decode(best_sequence)
        return decoded

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
        if not isinstance(self.beam_searcher, BeamSearcher):
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
        if not isinstance(json_path, str) or not json_path:
            self._json_path = ''
            self._eow_token = ''
            self._text_processor = None
            return
        if not isinstance(eow_token, str) or not eow_token:
            self._json_path = ''
            self._eow_token = ''
            self._text_processor = None
            return
        self._json_path = json_path
        self._eow_token = eow_token
        self._text_processor = TextProcessor(eow_token)

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
        if not isinstance(n_gram_size, int) or n_gram_size <= 0:
            return None
        if not isinstance(self._json_path, str) or not self._json_path:
            return None
        if not isinstance(self._text_processor, TextProcessor):
            return None
        file_content = ''
        file_obj = open(self._json_path, 'r', encoding='utf-8')
        file_content = file_obj.read()
        file_obj.close()
        if not isinstance(file_content, str) or not file_content:
            return None
        content = json.loads(file_content)
        if not check_dict(content, tuple, int, False):
            return None
        cleaned = {}
        for ngram, freq in content.items():
            if not isinstance(ngram, tuple) or len(ngram) != n_gram_size:
                continue
            valid_ngram = []
            for ch in ngram:
                if not isinstance(ch, str):
                    continue
                ch = ch.strip()
                if not ch.isalpha() and ch != ' ':
                    continue
                if ch == ' ':
                    ch = self._eow_token
                valid_ngram.append(ch)
            if len(valid_ngram) == n_gram_size:
                cleaned[tuple(valid_ngram)] = freq
        if not cleaned:
            return None
        self._text_processor.fill_from_ngrams(cleaned)
        encoded = []
        for ngram in cleaned:
            for letter in ngram:
                letter_id = self._text_processor.get_id(letter)
                if letter_id is not None:
                    encoded.append(letter_id)
        if not encoded:
            return None
        model = NGramLanguageModel(tuple(encoded), n_gram_size)
        model.set_n_grams(cleaned)
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
        if not isinstance(language_models, tuple) or not language_models:
            self._language_models = {}
            self._text_processor = None
            return
        if not all(isinstance(model, NGramLanguageModel) for model in language_models):
            self._language_models = {}
            self._text_processor = None
            return
        if not isinstance(text_processor, TextProcessor):
            self._language_models = {}
            self._text_processor = None
            return
        sorted_models = sorted(language_models, key=lambda m: m.get_n_gram_size(), reverse=True)
        self._language_models = {model.get_n_gram_size(): model for model in sorted_models}
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
        if not isinstance(self._text_processor, TextProcessor):
            return None
        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None  
        result = list(encoded_prompt)
        for _ in range(seq_len):
            next_probs = self._get_next_token(tuple(result))
            if not next_probs:
                break
            next_token = max(next_probs.items(), key=lambda x: (x[1], x[0]))[0]
            result.append(next_token)
            end_of_word_id = self._text_processor.get_id(
                self._text_processor.get_end_of_word_token()
            )
            if next_token == end_of_word_id:
                break
        decoded = self._text_processor.decode(tuple(result))
        return decoded

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
        if not self._language_models:
            return None
        for _, model in self._language_models.items():
            probs = model.generate_next_token(sequence_to_continue)
            if probs is None:
                continue
            if probs:
                return probs
            if len(sequence_to_continue) > 1:
                shorter_sequence = sequence_to_continue[1:]
                probs = model.generate_next_token(shorter_sequence)
                if probs:
                    return probs
        return None
