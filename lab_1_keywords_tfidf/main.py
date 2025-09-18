"""
Lab 1

Extract keywords based on frequency related metrics
"""

# pylint:disable=unused-argument
from typing import Any


def clean_and_tokenize(text: str) -> list[str] | None:
    if not isinstance(text, str):
        return None
    text = text.lower()
    cleaned_text = ''
    for word in text:
        if word.isalnum() or word.isspace():
            cleaned_text += word
    return cleaned_text.split()
    """
    Remove punctuation, convert to lowercase, and split into tokens.

    Args:
        text (str): Original text

    Returns:
        list[str] | None: A list of lowercase tokens without punctuation.
        In case of corrupt input arguments, None is returned.
    """


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
    if tokens is None or stop_words is None:
        return None
    if not isinstance(tokens, list) or not isinstance(stop_words, list):
        return None
    for token in tokens:
        if not isinstance(token, str):
            return None
    for words in stop_words:
        if not isinstance(words, str):
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
    if tokens is None:
        return None
    if not isinstance(tokens, list):
        return None
    frequency_dictionary = {}
    for token in tokens:
        if not isinstance(token, str):
            return None
        frequency_dictionary[token] = tokens.count(token)
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
    if not isinstance(frequencies, dict) or not isinstance(top, int):
        return None
    if not frequencies:
        return None
    if top < 1:
        return None
    for key, val in frequencies.items():
        if not isinstance(val, (int, float)) or not isinstance(key, str):
            return None
    if top > len(frequencies):
        top = len(frequencies)
    sorted_tokens = sorted(frequencies.keys(),
                          key=lambda x: (-frequencies[x], x))
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
    if not isinstance(frequencies, dict) or not frequencies:
        return None
    for val in frequencies.values():
        if not isinstance(val, int) or val < 0:
            return None
    for key in frequencies.keys():
        if not isinstance(key, str):
            return None
    total_tokens = sum(frequencies.values())
    if total_tokens == 0:
        return None
    return {token: val / total_tokens for token, val in frequencies.items()}


def calculate_tfidf(term_freq: dict[str, float], idf: dict[str, float]) -> dict[str, float] | None:
    if not isinstance(term_freq, dict) or not isinstance(idf, dict) or not term_freq:
        return None
    for key, val in term_freq.items():
        if not isinstance(key, str) or not isinstance(val, float):
            return None
    """
    Calculate TF-IDF score for tokens.

    Args:
        term_freq (dict[str, float]): Term frequency values
        idf (dict[str, float]): Inverse document frequency values

    Returns:
        dict[str, float] | None: Dictionary with tokens and TF-IDF values.
        In case of corrupt input arguments, None is returned.
    """


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
