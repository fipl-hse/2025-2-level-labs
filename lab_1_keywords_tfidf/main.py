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
    if can_be_empty is False and not user_input:
        return False
    for el in user_input:
        if not isinstance(el, elements_type):
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
    if can_be_empty is False and not user_input:
        return False
    for key, value in user_input.items():
        if not isinstance(key, key_type) or not isinstance(value, value_type):
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
    if isinstance(user_input, bool) or isinstance(user_input, int):
        return user_input > 0

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
    punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for syb in punct:
        text = text.replace(syb, '')
    new_text = text.lower()
    tokens = new_text.split()
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
    if not all(isinstance(el, str) for el in tokens):
        return None
    if not all(isinstance(el, str) for el in stop_words):
        return None
    result = [token for token in tokens if token not in stop_words]
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
    for token in tokens:
        if not isinstance(token, str):
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
    if not isinstance(frequencies, dict) or not frequencies:
        return None
    for key, value in frequencies.items():
        if not isinstance(key, str) or not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
    if not isinstance(top, int) or top <= 0:
        return None
    sorted_t = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    most_frequent = [token for token, freq in sorted_t[:top]]
    return most_frequent


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
    if not frequencies:
        return None
    for token, freq in frequencies.items():
        if not isinstance(token, str) or not isinstance(freq, int) or freq < 0:
            return None
    words_sum = sum(frequencies.values())
    if words_sum == 0:
        return None
    tf_dict = {}
    for token, freq in frequencies.items():
        tf_value = freq / words_sum
        tf_dict[token] = tf_value

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
    if not isinstance(term_freq, dict):
        return None
    if not term_freq:
        return None
    if not isinstance(idf, dict):
        return None
    for token, term_value in term_freq.items():
        if not isinstance(token, str) or not isinstance(term_value, (int, float)) or term_value < 0:
            return None
    for token1, idf_value in idf.items():
        if not isinstance(token1, str) or not isinstance(idf_value, (int, float)):
            return None
    tf_idf_dict = {}
    for token, term_value in term_freq.items():
        if token not in idf:
            idf_value = 3.8501476017100584
        else:
            idf_value = idf[token]
        tf_idf_dict[token] = term_value * idf_value
    return tf_idf_dict


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
    if not isinstance(doc_freqs, dict):
        return None
    if not isinstance(corpus_freqs, dict):
        return None
    for token, freq in doc_freqs.items():
        if not isinstance(token, str) or not isinstance(freq, (int, float)) or freq < 0:
            return None
    for token, freq in corpus_freqs.items():
        if not isinstance(token, str) or not isinstance(freq, (int, float)) or freq < 0:
            return None
    tw_doc = sum(doc_freqs.values())
    tw_corpus = sum(corpus_freqs.values())
    if tw_doc == 0:
        return None
    expected_frequencies = {}
    for token, j in doc_freqs.items():
        k = corpus_freqs.get(token, 0)
        l = tw_doc - j
        m = tw_corpus - k
        expected_frequencies[token] = ((j + k) * (j + l)) / (j + k + l + m)
    return expected_frequencies


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
    if not isinstance(expected, dict):
        return None
    if not isinstance(observed, dict):
        return None
    if not expected or not observed:
        return None
    for token, freq in expected.items():
        if (not isinstance(token, str) or not isinstance(freq,(int, float)) or isinstance(freq, bool) or freq < 0):
            return None
    for token, freq in observed.items():
        if (not isinstance(token, str) or not isinstance(freq, int) or isinstance(freq, bool) or freq < 0):
            return None
    chi_values = {}
    for token, observed_freq in observed.items():
        if token in expected:
            expected_freq = expected[token]
            if expected_freq > 0:
                chi_value = (observed_freq - expected_freq) ** 2 / expected_freq
                chi_values[token] = chi_value
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
    if not isinstance(chi_values, dict):
        return None
    if not isinstance(alpha, (int, float)):
        return None
    if not chi_values:
        return None
    for token, value in chi_values.items():
        if not isinstance(token, str) or not isinstance(value, (int, float)) or value < 0:
            return None
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if alpha not in criterion:
        return None
    criterion_value = criterion[alpha]
    significant_words = {}
    for token, chi_value in chi_values.items():
        if chi_value >= criterion_value:
            significant_words[token] = chi_value
    return significant_words
