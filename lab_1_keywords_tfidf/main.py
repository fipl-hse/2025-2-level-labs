"""
Lab 1

Extract keywords based on frequency related metrics
"""

from math import log

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

    if not user_input and can_be_empty:
        return False

    for item in user_input:
        if not isinstance(item, elements_type):
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

    excess_symbols = str.maketrans("", "", "!,.:…@!$&><*#%№()[]}{\"/*-_=;“”'|~`«»;—―?")
    cleaned_tokens = text.lower().translate(excess_symbols).split()
    return cleaned_tokens


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
    if not check_list(tokens, str, False) or not check_list(stop_words, str, False):
        return None

    cleansed_tokens = [token for token in tokens if token not in stop_words]
    return cleansed_tokens


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

    frequencies = {token: tokens.count(token) for token in tokens}
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
    if not (check_dict(frequencies, str, int, False) or check_dict(frequencies, str, float, False)):
        return None

    if not check_positive_int(top):
        return None

    return sorted(frequencies.keys(), key = lambda x: frequencies[x], reverse = True)[:top]


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

    doc_wordcount: int = sum(frequencies.values())
    term_freq = {token: count / doc_wordcount for token, count in frequencies.items()}
    return term_freq


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

    return {word: count * idf.get(word, log(47)) for word, count in term_freq.items()}


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

    doc_wordcount: int = sum(doc_freqs.values())
    corpus_wordcount: int = sum(corpus_freqs.values())
    expected_frequency: dict[str,float] = {}

    for token, value in doc_freqs.items():
        corpus_value = corpus_freqs.get(token,0)

        expected = (
            (value + corpus_value) * (value + (doc_wordcount - value))
            /
            (value
            + corpus_value
            + (doc_wordcount - value)
            + (corpus_wordcount - corpus_value))
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
    if not check_dict(expected, str, float, False) or not check_dict(observed, str, int, False):
        return None

    return {token: (observed[token] - value )** 2 / value for token, value in expected.items()}



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
    if not (check_dict(chi_values, str, float, False) or not check_float(alpha)):
        return None

    criterion = {0.05: 3.841458821, 0.01: 6.634896601, 0.001: 10.82756617}

    if not (threshold := criterion.get(alpha)):
        return None

    return {token: value for token, value in chi_values.items() if value > threshold}
