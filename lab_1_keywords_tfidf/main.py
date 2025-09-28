"""
Lab 1

Extract keywords based on frequency related metrics
"""

# pylint:disable=unused-argument
import math
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
    if not user_input and can_be_empty is False:
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
    if not user_input and can_be_empty is False:
        return False
    return all(isinstance(k, key_type) and isinstance(v, value_type)
               for k, v in user_input.items())


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
    cleaned_chars = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            cleaned_chars.append(ch)
    text_clean = ''.join(cleaned_chars)

    tokens = [token for token in text_clean.split() if token]

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
    if (
        not check_list(tokens, str, False)
        or not check_list(stop_words, str, True)
    ):
        return None

    return [t for t in tokens if t not in stop_words]


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
    if not frequencies:
        return None
    if not (
        check_dict(
            frequencies,
            str,
            int,
            True) or check_dict(
            frequencies,
            str,
            float,
            True)):
        return None
    if not check_positive_int(top):
        return None

    return [
        key for key,
        _ in sorted(
            frequencies.items(),
            key=lambda x: x[1],
            reverse=True)[
            :top]]


def calculate_tf(frequencies: dict[str, int]) -> dict[str, float] | None:
    """
    Calculate Term Frequency (TF) for each token.

    Args:
        frequencies (dict[str, int]): Raw occurrences of tokens

    Returns:
        dict[str, float] | None: Dictionary with tokens and TF values.
        In case of corrupt input arguments, None is returned.
    """
    if not frequencies or not check_dict(frequencies, str, int, True):
        return None

    total = sum(frequencies.values())
    if total == 0:
        return None

    return {k: v / total for k, v in frequencies.items()}


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
    if not term_freq:
        return None
    if not (
        check_dict(term_freq, str, float, True) and
        check_dict(idf, str, float, True)
    ):
        return None

    total_docs = 47
    tfidf_dict = {}

    for word, tf_value in term_freq.items():
        idf_value = idf.get(word)
        if idf_value is None:
            idf_value = log(total_docs / 1)
        tfidf_dict[word] = tf_value * idf_value

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

    if not doc_freqs:
        return None
    if not (check_dict(doc_freqs, str, int, True) and
            check_dict(corpus_freqs, str, int, True)):
        return None

    total_doc_words = sum(doc_freqs.values())
    total_corpus_words = sum(corpus_freqs.values())

    expected_freqs = {}
    for word in doc_freqs:
        doc_count = doc_freqs[word]
        corpus_count = corpus_freqs.get(word, 0)
        other_doc_words = total_doc_words - doc_count
        other_corpus_words = total_corpus_words - corpus_count

        expected_freqs[word] = (
            (doc_count + corpus_count) *
            (doc_count + other_doc_words) /
            (doc_count + corpus_count + other_doc_words + other_corpus_words)
        )

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
    if not expected or not observed:
        return None
    if not (
        check_dict(
            expected,
            str,
            float,
            True) and check_dict(
            observed,
            str,
            int,
            True)):
        return None

    chi_values = {}
    for term, exp_val in expected.items():
        obs_val = observed.get(term, 0)
        chi_values[term] = (obs_val - exp_val) ** 2 / \
            exp_val if exp_val != 0 else 0.0

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
    if not chi_values:
        return None
    if not check_dict(chi_values, str, float, True) or not check_float(alpha):
        return None

    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if alpha not in criterion:
        return None

    return {
        word: chi_value
        for word, chi_value in chi_values.items()
        if chi_value > criterion[alpha]
    }
