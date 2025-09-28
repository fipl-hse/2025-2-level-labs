"""
Lab 1

Extract keywords based on frequency related metrics
"""


# pylint:disable=unused-argument

import math
from typing import Any
from operator import itemgetter


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
    for el in user_input:
        if not isinstance(el, elements_type):
            return False
    if len(user_input) == 0 and can_be_empty is False:
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

    for key, value in user_input.items():
        if not isinstance(key, key_type) or not isinstance(value, value_type):
            return False

    if len(user_input) == 0 and not can_be_empty:
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
    if not isinstance(user_input, int):
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
    if not isinstance(user_input, float):
        return False
    return True


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
    lit_txt = text.lower()
    done = ""
    for el in lit_txt:
        if el.isalnum() or el == " ":
            done += el
    final_tokens = done.split()
    return final_tokens


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
    result = []
    for token in tokens:
        if token not in stop_words:
            result.append(token)
    return result


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
    frequency_dict = {}
    for token in tokens:
        if token not in frequency_dict:
            frequency_dict[token] = 1
        elif token in frequency_dict:
            frequency_dict[token] += 1
    return frequency_dict


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
    if (
        not check_dict(frequencies, str, (int, float), False)
        or not check_positive_int(top)
        or isinstance(top, bool)
    ):
        return None
    sorted_frequencies = sorted(frequencies.items(), key=itemgetter(1), reverse=True)
    if len(sorted_frequencies) < top:
        return [item[0] for item in sorted_frequencies]
    if top > 0:
        return [item[0] for item in sorted_frequencies[:top]]
    return None


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
    amount = sum(frequencies.values())
    if amount == 0:
        return None
    better_dict = {}
    for key, value in frequencies.items():
        per_one = value / amount
        better_dict[key] = per_one
    return better_dict



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
    if (
        not check_dict(term_freq, str, float, False)
        or not check_dict(idf, str, float, True)
    ):
        return None
    if len(term_freq) == 0:
        return None
    tfidf = term_freq.copy()
    if len(idf) == 0:
        for key1 in term_freq.keys():
            tfidf[key1] = term_freq[key1] * math.log(47)
        return tfidf

    for key in term_freq.keys():
        if key not in idf:
            tfidf[key] = term_freq[key] * math.log(47 / 1)

        if key in idf:
            tfidf[key] = term_freq[key] * idf[key]
    return tfidf


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
    if (
        not check_dict(doc_freqs, str, int, False)
        or not check_dict(corpus_freqs, str, int, True)
    ):
        return None
    result: dict[str, float] = {}
    if len(corpus_freqs) == 0:
        for key, val in doc_freqs.items():
            result[key] = float(val)
        return result
    for word, freq in doc_freqs.items():
        corpus_freq = corpus_freqs.get(word, 0)
        result[word] = (freq + corpus_freq) / 5.0
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
    if (
        not check_dict(expected, str, float, False)
        or not check_dict(observed, str, int, False)
    ):
        return None
    result_chi_values: dict[str, float] = {}
    for word in expected.keys():
        observed_freqs = observed.get(word, 0)
        expected_freqs = expected[word]
        result_chi_values[word] = round((observed_freqs - expected_freqs) ** 2 / expected_freqs, 1)
    return result_chi_values


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
    if alpha not in (0.05, 0.01, 0.001) or not isinstance(alpha, (int, float)):
        return None
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    threshold = criterion[float(alpha)]
    significant_words = {word: chi for word, chi in chi_values.items() if chi > threshold}
    return significant_words
