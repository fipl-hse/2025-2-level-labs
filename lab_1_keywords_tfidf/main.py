"""
Lab 1

Extract keywords based on frequency related metrics
"""

import math

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
    if not user_input and can_be_empty is False:
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
    if not user_input and can_be_empty is False:
        return False
    for k, v in user_input.items():
        if not isinstance(k, key_type) or not isinstance(v, value_type):
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
    if not isinstance(user_input,int) or user_input<=0 or isinstance(user_input, bool):
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
    return isinstance(user_input,float)

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
    text=text.lower()
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for el in text:
        if el in punctuation:
            text=text.replace(el, "")
    text_new=text.split(' ')
    return [el for el in text_new if el.strip()]

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
    return [el for el in tokens if el not in stop_words]

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
    return {token:tokens.count(token) for token in tokens}

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
    sorted_frequencies = sorted(frequencies.keys(), key = lambda x: frequencies[x], reverse=True)
    return sorted_frequencies[:top]

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
    total = sum(frequencies.values())
    if total==0:
        return {}
    tf_tokens={}
    for k,v in frequencies.items():
        tf_tokens[k] = v / total
    return dict(sorted(tf_tokens.items()))


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
    tf_idf={}
    for k, v in term_freq.items():
        tf_idf[k] = v*idf.get(k, math.log(47))
    return tf_idf

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
    exp_freq_dict={}
    sum_in_doc = sum(doc_freqs.values())
    sum_in_corpus = sum(corpus_freqs.values())
    exp_freq_dict={}
    for t, t_in_d in doc_freqs.items():
        t_in_D=corpus_freqs.get(t, 0)
        all_words_in_d=sum_in_doc-t_in_d
        all_word_in_D=sum_in_corpus-t_in_D
        multi=(t_in_d+t_in_D)*(t_in_d+all_words_in_d)
        sum_all_words=t_in_d+t_in_D+all_words_in_d+all_word_in_D
        if sum_all_words==0:
            exp_freq_dict[t] = 0.0
        else:
            exp_freq_dict[t]=float(multi/sum_all_words)
    return dict(sorted(exp_freq_dict.items()))

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
    return {word: pow(expected[word]-observed[word], 2) / expected[word] for word in expected}

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
    if not check_dict(chi_values, str, float, False) or not isinstance(alpha, float) or alpha not in criterion:
        return None
    critical_value=criterion[alpha]
    return {word: chi_values[word] for word in chi_values if chi_values[word]>critical_value}