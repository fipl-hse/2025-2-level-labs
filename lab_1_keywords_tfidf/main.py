"""
Lab 1

Extract keywords based on frequency related metrics
"""

from math import log

# pylint:disable=unused-argument
from typing import Any


def check_list(
        user_input: Any,
        elements_type: type | tuple[type, ...],
        can_be_empty: bool
        ) -> bool:
    """
    Check if the object is a list containing elements of a certain type.

    Args:
        user_input (Any): Object to check
        elements_type (type): Expected type of list elements
        can_be_empty (bool): Whether an empty list is allowed

    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(user_input, list):
        return False

    if not user_input and not can_be_empty:
        return False

    for element in user_input:
        if not isinstance(element, elements_type):
            return False

    return True


def check_dict(
        user_input: Any,
        key_type: type,
        value_type: type | tuple[type, ...],
        can_be_empty: bool
        ) -> bool:
    """
    Check if the object is a dictionary with keys and values of given types.

    Args:
        user_input (Any): Object to check
        key_type (type): Expected type of dictionary keys
        value_type (type): Expected type of dictionary values
        can_be_empty (bool): Whether an empty dictionary is allowed

    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(user_input, dict):
        return False

    if not user_input and not can_be_empty:
        return False

    for key, value in user_input.items():
        if not isinstance(key, key_type):
            return False
        if not isinstance(value, value_type):
            return False

    return True


def check_positive_int(user_input: Any) -> bool:
    """
    Check if the object is a positive integer (not bool).

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(user_input, int) or isinstance(user_input, bool):
        return False

    if user_input <= 0:
        return False

    return True


def check_float(user_input: Any) -> bool:
    """
    Check if the object is a float.

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """

    return isinstance(user_input, float)


def clean_and_tokenize(text: str) -> list[str] | None:
    """
    Remove punctuation, convert to lowercase, and split into tokens.

    Args:
        text (str): Original text

    Returns:
        list[str] | None: A list of lowercase tokens without punctuation.
        In case of corrupt input arguments, None is returned.
    """
    if not isinstance(text, str):
        return None

    forbidden_symbols: str = ":./?,! \"'-№#&*><;%@()}{[]|=$+-_\\~"
    tab = str.maketrans("", "", forbidden_symbols)

    cleaned_and_tokenized_text: list[str] = []

    for word in text.split():
        cleaned_word: str = word.translate(tab).lower()

        if cleaned_word:
            cleaned_and_tokenized_text.append(cleaned_word)

    return cleaned_and_tokenized_text


def remove_stop_words(tokens: list[str], stop_words: list[str]) -> list[str] | None:
    """
    Exclude stop words from the token sequence.

    Args:
        tokens (list[str]): Original token sequence
        stop_words (list[str]): Tokens to exclude

    Returns:
        list[str] | None: Token sequence without stop words.
        In case of corrupt input arguments, None is returned.
    """
    if not check_list(tokens, str, False):
        return None
    if not check_list(stop_words, str, True):
        return None

    return [word for word in tokens if word not in stop_words]


def calculate_frequencies(tokens: list[str]) -> dict[str, int] | None:
    """
    Create a frequency dictionary from the token sequence.

    Args:
        tokens (list[str]): Token sequence

    Returns:
        dict[str, int] | None: A dictionary {token: occurrences}.
        In case of corrupt input arguments, None is returned.
    """
    if not check_list(tokens, str, False):
        return None

    return {word: tokens.count(word) for word in set(tokens)}


def get_top_n(frequencies: dict[str, int | float], top: int) -> list[str] | None:
    """
    Extract the most frequent tokens.

    Args:

        frequencies (dict[str, int | float]): A dictionary with tokens and their frequencies
        top (int): Number of tokens to extract

    Returns:
        list[str] | None: Top-N tokens sorted by frequency.
        In case of corrupt input arguments, None is returned.
    """
    if not check_dict(frequencies, str, (int, float), False):
        return None
    if not check_positive_int(top):
        return None

    top_words: list[str] = []

    for element in sorted(frequencies.items(), key=lambda item: item[1], reverse=True):
        top_words.append(element[0])

    return top_words[:top]


def calculate_tf(frequencies: dict[str, int]) -> dict[str, float] | None:
    """
    Calculate Term Frequency (TF) for each token.

    Args:
        frequencies (dict[str, int]): Raw occurrences of tokens

    Returns:
        dict[str, float] | None: Dictionary with tokens and TF values.
        In case of corrupt input arguments, None is returned.
    """
    if not check_dict(frequencies, str, int, False):
        return None

    document_word_count: int = sum(frequencies.values())

    return {token: count / document_word_count for token, count in frequencies.items()}


def calculate_tfidf(term_freq: dict[str, float], idf: dict[str, float]) -> dict[str, float] | None:
    """
    Calculate TF-IDF score for tokens.

    Args:
        term_freq (dict[str, float]): Term frequency values
        idf (dict[str, float]): Inverse document frequency values

    Returns:
        dict[str, float] | None: Dictionary with tokens and TF-IDF values.
        In case of corrupt input arguments, None is returned.
    """
    if not check_dict(term_freq, str, float, False):
        return None
    if not check_dict(idf, str, float, True):
        return None

    max_idf = log(47 / 1)

    return {token: count * idf.get(token, max_idf) for token, count in term_freq.items()}


def calculate_expected_frequency(
    doc_freqs: dict[str, int], corpus_freqs: dict[str, int]
) -> dict[str, float] | None:
    """
    Calculate expected frequency for tokens based on document and corpus frequencies.

    Args:
        doc_freqs (dict[str, int]): Token frequencies in document
        corpus_freqs (dict[str, int]): Token frequencies in corpus

    Returns:
        dict[str, float] | None: Dictionary with expected frequencies.
        In case of corrupt input arguments, None is returned.
    """
    if not check_dict(doc_freqs, str, int, False):
        return None
    if not check_dict(corpus_freqs, str, int, True):
        return None

    document_word_count: int = sum(doc_freqs.values())
    corpus_word_count: int = sum(corpus_freqs.values())
    expected_frequency: dict[str, float] = {}

    for token, value in doc_freqs.items():
        value_in_corpus = corpus_freqs.get(token, 0)

        expected = (
            (value + value_in_corpus)
            * (value + (document_word_count - value))
            / (
                value
                + value_in_corpus
                + (document_word_count - value)
                + (corpus_word_count - value_in_corpus)
            )
        )

        expected_frequency[token] = expected

    return expected_frequency


def calculate_chi_values(
    expected: dict[str, float], observed: dict[str, int]
) -> dict[str, float] | None:
    """
    Calculate chi-squared values for tokens.

    Args:
        expected (dict[str, float]): Expected frequencies
        observed (dict[str, int]): Observed frequencies

    Returns:
        dict[str, float] | None: Dictionary with chi-squared values.
        In case of corrupt input arguments, None is returned.
    """
    if not check_dict(expected, str, float, False):
        return None
    if not check_dict(observed, str, int, False):
        return None

    return {token: (observed[token] - value) ** 2 / value for token, value in expected.items()}


def extract_significant_words(
    chi_values: dict[str, float], alpha: float
) -> dict[str, float] | None:
    """
    Select tokens with chi-squared values greater than the critical threshold.

    Args:
        chi_values (dict[str, float]): Dictionary with chi-squared values
        alpha (float): Significance level controlling chi-squared threshold

    Returns:
        dict[str, float] | None: Dictionary with significant tokens.
        In case of corrupt input arguments, None is returned.
    """
    if not check_dict(chi_values, str, float, False):
        return None
    if not check_float(alpha):
        return None

    calculated_alphas = {0.05: 3.841458821, 0.01: 6.634896601, 0.001: 10.82756617}

    threshold = calculated_alphas.get(alpha)
    if not (threshold := calculated_alphas.get(alpha)):
        return None

    return {token: value for token, value in chi_values.items() if value > threshold}
