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
    if not user_input:
        return can_be_empty
    if not isinstance(user_input, list):
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
    if not user_input:
        return can_be_empty
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
    text_clean_and_tokenized = []
    for raw in text.split():
        clean_word = (''.join(symbol.lower() for symbol in raw if symbol.isalnum()))
        if clean_word:
            text_clean_and_tokenized.append(clean_word)
    return text_clean_and_tokenized

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
    if not check_list(tokens, str, False):
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
    if not check_list(tokens, str, False):
        return None
    return {token: tokens.count(token) for token in tokens}

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
    if (not check_dict(frequencies, str, int, False) and
        not check_dict(frequencies, str, float, False)) or not check_positive_int(top):
        return None
    freq_lst_sorted = sorted(frequencies.items(), key = lambda item: (-item[1], item[0]))
    top = min(top, len(freq_lst_sorted))
    top_words = [item[0] for item in freq_lst_sorted[:top]]
    return top_words

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
    total_tokens_count = sum(frequencies.values())
    return {tok: token_value / total_tokens_count for tok, token_value in frequencies.items()}

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

    result = {}
    for token, tok_freq_value in term_freq.items():
        idf_val = idf.get(token, math.log(47))
        result[token] = tok_freq_value * idf_val
    return result

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
    expected_freq = {}
    total_doc_freqs = sum(doc_freqs.values())
    total_corpus_freqs = sum(corpus_freqs.values())

    for token, freq in doc_freqs.items():
        in_doc_tok_count = freq
        in_corpus_tok_count = corpus_freqs.get(token, 0)
        doc_without_tok = total_doc_freqs - in_doc_tok_count
        corpus_without_tok = total_corpus_freqs - in_corpus_tok_count
        expected_value = (in_doc_tok_count + in_corpus_tok_count)*(in_doc_tok_count +
                            doc_without_tok)/(in_doc_tok_count +
                            in_corpus_tok_count + doc_without_tok + corpus_without_tok)
        expected_freq[token] = expected_value
    return expected_freq

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
    for token, exp_token_freq in expected.items():
        chi = (observed[token] - exp_token_freq)**2 / exp_token_freq
        chi_values[token] = chi
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
    thresholds = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if alpha not in thresholds:
        return None
    critical_value = thresholds[alpha]
    crit_value = thresholds[alpha]
    return {token: value for token, value in chi_values.items() if value > crit_value}