"""
Lab 1

Extract keywords based on frequency related metrics
"""

# pylint:disable=unused-argument
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
    if not can_be_empty and len(user_input) == 0:
        return False
    return all(isinstance(elem, elements_type) for elem in user_input)


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
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    cleaned_text = "".join(symbol for symbol in text if symbol not in punctuation)
    lower_cleaned_text = cleaned_text.lower()
    tokens = lower_cleaned_text.split()
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
    if not check_list(tokens, str, True) or not check_list(stop_words, str, True):
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
    if not check_list(tokens, str, True):
        return None
    return {token: tokens.count(token) for token in set(tokens)}


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
    if not isinstance(frequencies, dict):
        return None
    if not frequencies:
        return None
    if not all(isinstance(k, str) for k in frequencies):
        return None
    if not all(isinstance(v, (int, float)) for v in frequencies.values()):
        return None
    if not check_positive_int(top):
        return None
    sorted_keys = sorted(frequencies, key=lambda k: frequencies[k], reverse=True)
    return sorted_keys[:top] if top < len(sorted_keys) else sorted_keys


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
    total_words = sum(frequencies.values())
    if total_words == 0:
        return None
    return {k: round(v / total_words, 4) for k, v in frequencies.items()}


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
    if not check_dict(term_freq, str, float, True) or not check_dict(idf, str, float, True):
        return None
    if not term_freq:
        return None
    default_idf = math.log(47)
    tfidf_dict = {}
    for word in term_freq:
        tfidf_value = term_freq[word] * idf.get(word, default_idf)
        tfidf_dict[word] = tfidf_value
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
    if not check_dict(doc_freqs, str, int, True) or not check_dict(corpus_freqs, str, int, True):
        return None
    if not doc_freqs:
        return None
    doc_total = sum(doc_freqs.values())
    corpus_total = sum(corpus_freqs.values())
    if doc_total == 0 and corpus_total == 0:
        return None
    expected_freqs = {}
    for token in sorted(doc_freqs.keys()):
        token_doc_freq = doc_freqs[token]
        token_corpus_freq = corpus_freqs.get(token, 0)
        expected = (token_doc_freq + token_corpus_freq) * doc_total / (doc_total + corpus_total)
        expected_freqs[token] = expected
    return expected_freqs


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
    if not check_dict(expected, str, float, True) or not check_dict(observed, str, int, True):
        return None
    if not expected or not observed:
        return None
    chi_values = {}
    for token in observed:
        if token not in expected:
            continue
        observed_freq = observed[token]
        expected_freq = expected[token]
        if expected_freq > 0:
            chi_value = (observed_freq - expected_freq) ** 2 / expected_freq
            chi_values[token] = round(chi_value, 1)
        else:
            chi_values[token] = float('inf') if observed_freq > 0 else 0.0
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
    if not check_dict(chi_values, str, float, True) or not check_float(alpha):
        return None
    if not chi_values:
        return None
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if alpha not in criterion:
        return None
    critical_value = criterion[alpha]
    significant_words = {}
    for token, chi_value in chi_values.items():
        if chi_value > critical_value:
            significant_words[token] = chi_value
    return significant_words