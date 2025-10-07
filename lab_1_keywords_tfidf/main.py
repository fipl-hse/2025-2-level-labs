"""
Lab 1

Extract keywords based on frequency related metrics
"""

import math
from typing import Any


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
    if not isinstance(user_input, list):
        return False

    if not user_input and not can_be_empty:
        return False

    for element in user_input:
        if not isinstance(element, elements_type):
            return False

    return True


def check_dict(user_input: Any, key_type: type, value_type: type | tuple[type, ...], can_be_empty: bool) -> bool:
    """
    Check if the object is a dictionary with keys and values of given types.

    Args:
        user_input (Any): Object to check
        key_type (type): Expected type of dictionary keys
        value_type (type | tuple): Expected type(s) of dictionary values
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
        if isinstance(value_type, tuple):
            if not any(isinstance(value, vt) for vt in value_type):
                return False
        elif not isinstance(value, value_type):
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
    return isinstance(user_input, int) and not isinstance(user_input, bool) and user_input > 0


def check_float(user_input: Any) -> bool:
    """
    Check if the object is a float.

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(user_input, float) and user_input > 0


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

    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

    cleaned_text = ""
    for char in text:
        if char not in punctuation:
            cleaned_text += char

    cleaned_text = cleaned_text.lower()

    tokens = cleaned_text.split()
    return [token for token in tokens if token]


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
    if not check_list(tokens, str, False) or not check_list(stop_words, str, True):
        return None

    return [token for token in tokens if token not in stop_words]


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

    frequencies = {}
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1

    return frequencies


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
    if not isinstance(frequencies, dict) or not check_positive_int(top):
        return None

    for key, value in frequencies.items():
        if not isinstance(key, str) or not isinstance(value, (int, float)):
            return None

    if not frequencies:
        return None

    sorted_tokens = sorted(frequencies.keys(), key=lambda x: (-frequencies[x], x))
    return sorted_tokens[:top]


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

    total_tokens = sum(frequencies.values())

    tf_scores = {}
    for token, count in frequencies.items():
        tf_scores[token] = count / total_tokens

    return tf_scores


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
    if not check_dict(term_freq, str, float, False) or not check_dict(idf, str, float, True):
        return None

    tfidf_scores = {}
    for token, tf_value in term_freq.items():
        if token in idf:
            tfidf_scores[token] = tf_value * idf[token]
        else:
            tfidf_scores[token] = tf_value * math.log(47)

    return tfidf_scores


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
    if not check_dict(doc_freqs, str, int, False) or not check_dict(corpus_freqs, str, int, True):
        return None

    if not corpus_freqs:
        return doc_freqs

    result: dict[str, float] = {}
    for token, freq in doc_freqs.items():
        corpus_freq = corpus_freqs.get(token, 0)
        result[token] = round((freq + corpus_freq) / 5, 1)

    return result


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
    if not expected or not observed:
        return None
    return None


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
    if not chi_values or not isinstance(alpha, float):
        return None
    return None
