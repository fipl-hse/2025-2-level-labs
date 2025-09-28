"""
Lab 1

Extract keywords based on frequency related metrics
"""


from typing import Any
import string
import math

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
    return isinstance(user_input, float) and user_input > 0

def check_float(user_input: Any) -> bool:
    """
    Check if the object is a float.

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(user_input, float) and not isinstance(user_input, bool)

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
    
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = text.translate(translator).lower()
    
    tokens = cleaned_text.split()
    return [token for token in tokens if token]

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
    if not check_dict(frequencies, str, (int, float), True) or not check_positive_int(top):
        return None
    
    if len(frequencies) == 0:
        return None  
    
    sorted_items = sorted(frequencies.items(), key=lambda x: (-x[1], x[0]))
    return [item[0] for item in sorted_items[:top]]


def calculate_tf(frequencies: dict[str, int]) -> dict[str, float] | None:
    """
    Calculate Term Frequency (TF) for each token.

    Args:
        frequencies (dict[str, int]): Raw occurrences of tokens

    Returns:
        dict[str, float] | None: Dictionary with tokens and TF values.
        In case of corrupt input arguments, None is returned.
    """
    if not check_dict(frequencies, str, int, True):
        return None
    
    total_tokens = sum(frequencies.values())
    if total_tokens == 0:
        return {}
    
    tf_scores = {}
    for token, count in frequencies.items():
        tf_scores[token] = count / total_tokens
    
    return tf_scores


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
    
    if len(term_freq) == 0:
        return None
    
    tfidf_scores = {}
    for token, tf_value in term_freq.items():
        if token in idf:
            tfidf_scores[token] = tf_value * idf[token]
        else:
            tfidf_scores[token] = tf_value * math.log(47 / 1)
    
    return tfidf_scores


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

    total_doc_words = sum(doc_freqs.values())
    total_corpus_words = sum(corpus_freqs.values()) if corpus_freqs else 0

    result: dict[str, float] = {}

    for token, freq in doc_freqs.items():
        corpus_freq = corpus_freqs.get(token, 0) if corpus_freqs else 0

        j = freq
        k = corpus_freq
        l = total_doc_words - freq
        m = total_corpus_words - corpus_freq

        denominator = j + k + l + m
        if denominator == 0:
            expected = 0.0
        else:
            expected = ((j + k) * (j + l)) / denominator

        result[token] = round(expected, 1)

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
