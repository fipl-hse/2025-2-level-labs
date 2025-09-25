"""
Lab 1

Extract keywords based on frequency related metrics
"""

# pylint:disable=unused-argument
from typing import Any
from math import log


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
    
    if not can_be_empty and not user_input:
        return False
    
    for element in user_input:
        if not isinstance(element, elements_type):
            return False
    return True


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
    
    if not can_be_empty and not user_input:
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
    
    text_without_punctuation = ""

    for char in text:
        if char.isalnum() or char.isspace():
            text_without_punctuation += char

    text_cleaned = text_without_punctuation.lower()
    return text_cleaned.split()


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
    return [element for element in tokens if element not in stop_words]



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
    
    frequencies_dictionary = {}
    for token in tokens:
        frequencies_dictionary[token] = tokens.count(token)
    return frequencies_dictionary


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
    if not check_dict(frequencies, str, int | float, False) or not check_positive_int(top):
        return None

    tokens_sorted = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
    top_tokens = [element[0] for element in tokens_sorted]
    if len(top_tokens) > top:
        top_tokens = top_tokens[:top]
    return top_tokens


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
    
    tf_dict = dict()
    words_total = sum(frequencies.values())
    for key in frequencies:
        tf = frequencies[key] / words_total
        tf_dict[key] = tf
    return tf_dict


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
    
    tfidf = dict()
    for key in term_freq:
        if key in idf.keys():
            tfidf_element = idf[key] * term_freq[key]
            tfidf[key] = tfidf_element
        else:
            tfidf[key] = term_freq[key] * log(47/1)
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
    if not check_dict(doc_freqs, str, int, False) or not check_dict(corpus_freqs, str, int, True):
        return None
    expected_freqency_dictionary = dict()
    doc_total = sum(doc_freqs.values())
    corpus_total = sum(corpus_freqs.values())
    for key in doc_freqs:
        j = doc_freqs[key]
        l = doc_total - doc_freqs[key]
        if key in corpus_freqs:
            k = corpus_freqs[key]
            m = corpus_total - corpus_freqs[key]
            expected_freqency = (j + k) * (j + l) / (j + k + l + m)
            expected_freqency_dictionary[key] = expected_freqency
        else:
            expected_freqency = j * (j + l) / (j + l + corpus_total)
            expected_freqency_dictionary[key] = expected_freqency
    return expected_freqency_dictionary


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
    chi_values = dict()
    for key in expected:
        if key in observed.keys():
            chi_squared = (observed[key] - expected[key]) ** 2 / expected[key]
            chi_values[key] = chi_squared
        else:
            chi_squared = expected[key] ** 2 / expected[key]
            chi_values[key] = chi_squared
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
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if not check_dict(chi_values, str, float, False) or not check_float(alpha):
        return None
    if alpha not in criterion.keys():
        return None
    threshold = criterion[alpha]
    return {token: value for token, value in chi_values.items() if value > threshold}
