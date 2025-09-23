"""
Lab 1

Extract keywords based on frequency related metrics
"""

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
    for el in user_input:
        if not isinstance(el, elements_type):
            return False
    if len(user_input) == 0 and can_be_empty is False:
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
    tokens = done.split()
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
    if not isinstance(tokens, list) or not isinstance(stop_words, list):
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
    if not isinstance(tokens, list):
        return None
    if not all(isinstance(token, str) for token in tokens):
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
        not isinstance(frequencies, dict)
        or not isinstance(top, int)
        or len(frequencies) == 0
        or isinstance(top, bool)
    ):
        return None
    sorted_frequencies = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
    if len(sorted_frequencies) < top:
        return [item[0] for item in sorted_frequencies]
    if top > 0:
        return [item[0] for item in sorted_frequencies[:top]]
    else:
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
    if not isinstance(frequencies, dict):
        return None
    if not all(isinstance(key, str) for key in frequencies.keys()):
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
    import math

    if (
        not isinstance(term_freq, dict)
        or not isinstance(idf, dict)
        or not all(isinstance(word, str) for word in term_freq.keys())
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
            continue
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
    if not isinstance(doc_freqs, dict) or not isinstance(corpus_freqs, dict) or len(doc_freqs) == 0:
        return None
    for key1, value1 in doc_freqs.items():
        if not isinstance(key1, str) or not isinstance(value1, int) or isinstance(value1, bool):
            return None
    for key2, value2 in corpus_freqs.items():
        if not isinstance(key2, str) or not isinstance(value2, int) or isinstance(value2, bool):
            return None
    result = doc_freqs.copy()
    if len(corpus_freqs) == 0:
        for keys in result.keys():
            result[keys] = result[keys]
        return result
    for word, freq in doc_freqs.items():
        corpus_freq = corpus_freqs.get(word, 0)
        result[word] = round((freq + corpus_freq) / 5, 1)
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
    if not isinstance(expected, dict) or not isinstance(observed, dict) or expected == {} or observed == {}:
        return None
    for k, v in expected.items():
        if not isinstance(k, str) or not isinstance(v, float):
            return None
    for k1, v1 in observed.items():
        if not isinstance(k1, str) or not isinstance(v1, int):
            return None
    result_chi_values: dict[str, float] = {}
    for word in expected.keys():
        observed_freqs = observed.get(word, 0)
        expected_freqs = expected.get(word, 0)
        result_chi_values[word] = round((observed_freqs - expected_freqs)** 2 / expected_freqs, 1 )
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
    if not isinstance(chi_values, dict) or len(chi_values) == 0:
        return None
    if alpha not in (0.05, 0.01, 0.001) or not isinstance(alpha,(int, float)):
        return None
    for k,v in chi_values.items():
        if not isinstance(k, str) or not isinstance(v, float) or isinstance(v, bool):
            return None
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    threshold = criterion[float(alpha)]
    significant_words = {word: chi for word, chi in chi_values.items() if chi > threshold}
    return significant_words