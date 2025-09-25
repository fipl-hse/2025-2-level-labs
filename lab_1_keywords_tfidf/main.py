"""
Lab 1

Extract keywords based on frequency related metrics
"""
import math

# pylint:unused-argument
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
    if user_input == []:
        return can_be_empty
    if isinstance(user_input, list):
        for element in user_input:
            if not isinstance(element, elements_type):
                return False
        return True
    return False


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
    if user_input == {}:
        return can_be_empty
    if isinstance(user_input, dict):
        for key, value in user_input.items():
            if not isinstance(key, key_type) or not isinstance(value, value_type):
                return False
        return True
    return False


def check_positive_int(user_input: Any) -> bool:
    """
    Check if the object is a positive integer (not bool).

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """
    if isinstance(user_input, int) and not isinstance(user_input, bool):
        return user_input > 0
    return False


def check_float(user_input: Any) -> bool:
    """
    Check if the object is a float.

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(user_input, float)


def clean_and_tokenize(raw_text: str) -> list[str] | None:
    """
    Remove punctuation, convert to lowercase, and split into tokens.

    Args:
        text (str): Original text

    Returns:
        list[str] | None: A list of lowercase tokens without punctuation.
        In case of corrupt input arguments, None is returned.
    """
    if isinstance(raw_text, str):
        symbols_to_delete = '.,?-:;!%><#@$^&*()_'
        raw_text = raw_text.replace('\n', ' ').lower()
        for symbol in raw_text:
            if symbol in symbols_to_delete:
                raw_text = raw_text.replace(symbol, '')
        return raw_text.split()
    return None


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
    if check_list(tokens, str, True) and check_list(stop_words, str, True):
        for stop_word in stop_words:
            while stop_word in tokens:
                tokens.remove(stop_word)
        return tokens
    return None


def calculate_frequencies(tokens: list[str]) -> dict[str, int] | None:
    """
    Create a frequency dictionary from the token sequence.

    Args:
        tokens (list[str]): Token sequence

    Returns:
        dict[str, int] | None: A dictionary {token: occurrences}.
        In case of corrupt input arguments, None is returned.
    """
    if check_list(tokens, str, True):
        frequencies = {token: tokens.count(token) for token in tokens}
        return frequencies
    return None


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
    if (check_dict(frequencies, str, int, False) or check_dict(frequencies, str, float, False)) \
        and check_positive_int(top):
        return [item[0] for item in sorted(list(frequencies.items()), \
            key=lambda item: (-item[1], item[0]))][0:top]
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
    if check_dict(frequencies, str, int, False):
        return {
            token: frequencies[token] / sum(frequencies.values()) for token in frequencies.keys()
        }
    return None


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
    if check_dict(term_freq, str, float, False) and check_dict(idf, str, float, True):
        return {
            token: term_freq[token] * idf.get(token, math.log(47)) for token in term_freq.keys()
        }
    return None


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
    if check_dict(doc_freqs, str, int, False) and check_dict(corpus_freqs, str, int, True):
        return {
            token: (doc_freqs[token] + corpus_freqs.get(token, 0)) * \
                sum(doc_freqs.values()) / (sum(doc_freqs.values()) + sum(corpus_freqs.values())) \
                    for token in doc_freqs.keys()
        }
    return None


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
    if check_dict(expected, str, float, False) and check_dict(observed, str, int, False):
        return {
            token: (observed.get(token, 0) - expected[token]) ** 2 / expected[token]\
                for token in expected.keys()
        }
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
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    significant_words = {}
    if check_dict(chi_values, str, float, False) and alpha in criterion:
        for token in chi_values:
            if chi_values[token] > criterion[alpha]:
                significant_words[token] = chi_values[token]
        return significant_words
    return None
