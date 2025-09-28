"""
Lab 1

Extract keywords based on frequency related metrics
"""

# pylint:disable=unused-argument
from typing import Any
import string
import math


def check_list(user_input: Any, elements_type: type, can_be_empty: bool) -> bool:
    """
    Check if the object is a list containing elements of a certain type.

    Args:
        user_input (Any): Object to check
        elements_type (type): Expected type of list elements
        can_be_empty (bool): Whether an empty list is allowed

    Returns:
        bool: True if valid, False otherwise
    """
    if not user_input:
        return can_be_empty
    if not isinstance(user_input, list):
        return False
    if not can_be_empty and len(user_input) == 0:
        return False
    return all(isinstance(el, elements_type) for el in user_input)


def check_dict(user_input: Any, key_type: type, value_type: type, can_be_empty: bool) -> bool:
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
    if not can_be_empty and len(user_input) == 0:
        return False
    return all(isinstance(k, key_type) and isinstance(v, value_type) for k, v in user_input.items())


def check_positive_int(user_input: Any) -> bool:
    """
    Check if the object is a positive integer (not bool).

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(user_input, int) and not isinstance(user_input, bool) and user_input > 0


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
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return tokens


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
    if not (check_list(tokens, str, True) and check_list(stop_words, str, True)):
        return None
    stop_set = set(stop_words)
    return [t for t in tokens if t not in stop_set]


def calculate_frequencies(tokens: list[str]) -> dict[str, int] | None:
    """
    Create a frequency dictionary from the token sequence.

    Args:
        tokens (list[str]): Token sequence

    Returns:
        dict[str, int] | None: A dictionary {token: occurrences}.
        In case of corrupt input arguments, None is returned.
    """
    if not check_list(tokens, str, True):
        return None
    freqs: dict[str, int] = {}
    for token in tokens:
        freqs[token] = freqs.get(token, 0) + 1
    return freqs


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
    if not (check_dict(frequencies, str, (int | float), True) and check_positive_int(top)):
        return None
    sorted_tokens = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in sorted_tokens[:top]]


def calculate_tf(frequencies: dict[str, int]) -> dict[str, float] | None:
    """
    Calculate Term Frequency (TF) for each token.

    Args:
        frequencies (dict[str, int]): Raw occurrences of tokens

    Returns:
        dict[str, float] | None: Dictionary with tokens and TF values.
        In case of corrupt input arguments, None is returned.
    """
    if not check_dict(frequencies, str, int, True):
        return None
    total = sum(frequencies.values())
    if total == 0:
        return {}
    return {token: count / total for token, count in frequencies.items()}


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
    if not (check_dict(term_freq, str, float, True) and check_dict(idf, str, float, True)):
        return None
    result: dict[str, float] = {}
    for token, tf_val in term_freq.items():
        idf_val = idf.get(token, math.log(47))
        result[token] = tf_val * idf_val
    return result


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
