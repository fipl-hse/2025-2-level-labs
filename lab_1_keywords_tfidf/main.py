"""
Lab 1

Extract keywords based on frequency related metrics
"""

import math

# pylint:disable=unused-argument
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
    if not can_be_empty and len(user_input) == 0:
        return False
    return all(isinstance(element, elements_type) for element in user_input)


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
    return (isinstance(user_input, int) and not isinstance(user_input, bool)
            and user_input > 0)


def check_float(user_input: Any) -> bool:
    """
    Check if the object is a float.
    Args:
        user_input (Any): Object to check
    Returns:
        bool: True if valid, False otherwise
    """
    if isinstance(user_input, float):
        return True
    return False


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
    cleaned_text = ''
    for word in text:
        if word.isalnum() or word.isspace():
            cleaned_text += word
    return cleaned_text.split()


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
    if not check_list(tokens, str, True) or not check_list(stop_words, str, True):
        return None
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


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
    frequency_dictionary = {}
    for token in tokens:
        frequency_dictionary[token] = frequency_dictionary.get(token, 0) + 1
    return frequency_dictionary


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
    if not all([isinstance(frequencies, dict), check_positive_int(top), frequencies]):
        return None
    top = min(top, len(frequencies))
    sorted_tokens = sorted(frequencies.keys(),
                          key=lambda key: (-frequencies[key], key))
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
    return {token: val / total_tokens for token, val in frequencies.items()}


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
    tfidf_dict = {}
    for token, tf_value in term_freq.items():
        tfidf_dict[token] = tf_value * idf.get(token, math.log(47 / 1))
    return tfidf_dict


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
    expected_frequency = {}
    total_doc_freq = sum(doc_freqs.values())
    total_corp_freq = sum(corpus_freqs.values())
    for token in doc_freqs:
        t_in_doc = doc_freqs[token]
        t_in_corp = corpus_freqs.get(token, 0)
        without_t_in_doc = total_doc_freq - t_in_doc
        without_t_in_corp = total_corp_freq - t_in_corp
        expected = (((t_in_doc + t_in_corp) * (t_in_doc + without_t_in_doc))
                / (t_in_doc + t_in_corp + without_t_in_doc + without_t_in_corp))
        expected_frequency[token] = expected
    return dict(sorted(expected_frequency.items()))


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
    if not check_dict(expected, str, float, False) or not check_dict(observed, str, int, False):
        return None
    chi_values = {}
    for token in observed:
        chi_values[token] = ((observed[token] - expected[token]) ** 2) / expected[token]
    return chi_values


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
    if not check_dict(chi_values, str, float, False) or not check_float(alpha):
        return None
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if alpha not in criterion:
        return None
    significant_words = {}
    for key, val in chi_values.items():
        if val > criterion[alpha]:
            significant_words[key] = val
    return significant_words
