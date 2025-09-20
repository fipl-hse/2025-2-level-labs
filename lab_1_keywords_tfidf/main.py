"""
Lab 1

Extract keywords based on frequency related metrics
"""

# pylint:disable=unused-argument
from typing import Any
import math


def check_list(user_input: Any, elements_type: type, can_be_empty: bool) -> bool:
    if not user_input:
        return can_be_empty
    if not isinstance(user_input, list):
        return False
    for element in elements_type:
        if not isinstance(element, elements_type):
            return False
    return True
    """
    Check if the object is a list containing elements of a certain type.

    Args:
        user_input (Any): Object to check
        elements_type (type): Expected type of list elements
        can_be_empty (bool): Whether an empty list is allowed

    Returns:
        bool: True if valid, False otherwise
    """


def check_dict(user_input: Any, key_type: type, value_type: type, can_be_empty: bool) -> bool:
    if not user_input:
        return can_be_empty
    if not isinstance(user_input, dict):
        return False
    for key, value in user_input:
        if not isinstance(key,key_type) or not isinstance(value, value_type):
            return False
    return True
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


def check_positive_int(user_input: Any) -> bool:
    if not isinstance(user_input, int) or user_input < 0:
        return False
    return True
    """
    Check if the object is a positive integer (not bool).

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """


def check_float(user_input: Any) -> bool:
    if not isinstance(user_input, float):
        return False
    return True
    """
    Check if the object is a float.

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """


def clean_and_tokenize(text: str) -> list[str] | None:
    if not isinstance(text, str):
        return None
    lowed = text.lower()
    cleaned = ""
    for symbol in lowed:
        if symbol.isalnum() or symbol == " ":
            cleaned += symbol
    tokenized = cleaned.split()
    return tokenized
    """
    Remove punctuation, convert to lowercase, and split into tokens.

    Args:
        text (str): Original text

    Returns:
        list[str] | None: A list of lowercase tokens without punctuation.
        In case of corrupt input arguments, None is returned.
    """


def remove_stop_words(tokens: list[str], stop_words: list[str]) -> list[str] | None:
    if not isinstance (tokens, list) or not isinstance (stop_words, list):
        return None
    without_stop_words = []
    for token in tokens:
        if token not in stop_words:
            without_stop_words.append(token)
    return without_stop_words

    """
    Exclude stop words from the token sequence.

    Args:
        tokens (list[str]): Original token sequence
        stop_words (list[str]): Tokens to exclude

    Returns:
        list[str] | None: Token sequence without stop words.
        In case of corrupt input arguments, None is returned.
    """


def calculate_frequencies(tokens: list[str]) -> dict[str, int] | None:
    if not isinstance(tokens, list):
        return None
    for token in tokens:
        if type(token) != str:
            return None
    frequencies = {}
    for word in tokens:
        frequencies[word] = tokens.count(word)
    return frequencies


    """
    Create a frequency dictionary from the token sequence.

    Args:
        tokens (list[str]): Token sequence

    Returns:
        dict[str, int] | None: A dictionary {token: occurrences}.
        In case of corrupt input arguments, None is returned.
    """


def get_top_n(frequencies: dict[str, int | float], top: int) -> list[str] | None:
    if not isinstance(frequencies, dict) or not isinstance(top, int) or isinstance (top, bool):
        return None
    if frequencies == {} or top <= 0:
        return None
    sorted_freq_by_value = dict(sorted(frequencies.items(), key = lambda item: item[1], reverse = True))
    sorted_keys = list(sorted_freq_by_value.keys())
    top_n = sorted_keys[:top]
    return top_n                 
    """
    Extract the most frequent tokens.

    Args:

        frequencies (dict[str, int | float]): A dictionary with tokens and their frequencies
        top (int): Number of tokens to extract

    Returns:
        list[str] | None: Top-N tokens sorted by frequency.
        In case of corrupt input arguments, None is returned.
    """


def calculate_tf(frequencies: dict[str, int]) -> dict[str, float] | None:
    if not isinstance(frequencies, dict):
        return None
    for key, value in frequencies.items():
        if type(key) != str or type(value) != int:
            return None
    all_words = sum(list(frequencies.values()))
    term_freq = {}
    for word, value in frequencies.items():
        term_freq[word] = value / all_words
    return term_freq




    """
    Calculate Term Frequency (TF) for each token.

    Args:
        frequencies (dict[str, int]): Raw occurrences of tokens

    Returns:
        dict[str, float] | None: Dictionary with tokens and TF values.
        In case of corrupt input arguments, None is returned.
    """


def calculate_tfidf(term_freq: dict[str, float], idf: dict[str, float]) -> dict[str, float] | None:
    if not isinstance(term_freq, dict) or not isinstance(idf, dict):
        return None
    for key in term_freq:
        if not isinstance(key, str):
            return None
    if term_freq == {}:
        return None
    tfidf_dict = {}
    if idf == {}:
        for term, freq in term_freq.items():
            tfidf_dict[term] = freq * math.log(47 / 1)
        return tfidf_dict
    for term, freq in term_freq.items():
        if term in idf:
           tfidf_dict[term] = freq * idf[term]
        else:
            tfidf_dict[term] = freq * math.log(47 / 1)
    return tfidf_dict
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
    if not isinstance(doc_freqs, dict) or not isinstance(corpus_freqs, dict):
        return None
    for key in doc_freqs:
        if not isinstance(key, str):
            return None
    if doc_freqs == {}:
        return None
    expected_frequency = {}
    words_in_doc = sum(doc_freqs.values())
    words_in_corpus = sum(corpus_freqs.values())
    for word in doc_freqs:
        word_in_doc = doc_freqs[word]
        word_in_corpus = corpus_freqs.get(word, 0)
        wo_word_in_doc = words_in_doc - word_in_doc
        wo_word_in_corpus = words_in_corpus - word_in_corpus
        expected = (
            (word_in_doc + word_in_corpus)*(word_in_doc + wo_word_in_doc)/(word_in_doc + word_in_corpus + wo_word_in_doc + wo_word_in_corpus)
            )
        expected_frequency[word] = expected
    return dict(sorted(expected_frequency.items()))

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
    if not isinstance(expected, dict) or not isinstance(observed, dict):
        return None
    if expected == {} or observed == {}:
        return None
    for key in expected:
        if not isinstance(key,str):
            return None
        if not isinstance(expected[key], (int, float)) or isinstance(expected[key], bool):
            return None
    for key in observed:
        if not isinstance(key,str):
            return None
        if not isinstance(observed[key], (int, float)) or isinstance(observed[key], bool):
            return None
    chi_values = {}
    for word in observed:
        chi_values[word] = (((observed[word] - expected[word])** 2) / expected[word])
    return chi_values
        
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
    if not isinstance(chi_values, dict) or not isinstance(alpha, float):
        return None
    if chi_values == {}:
        return None
    for key in chi_values:
        if not isinstance(key, str):
            return None
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if alpha not in criterion:
        return None
    significant_words = {}
    for key, value in chi_values.items():
        if value > criterion[alpha]:
            significant_words[key] = value
    return significant_words
    """
    Select tokens with chi-squared values greater than the critical threshold.

    Args:
        chi_values (dict[str, float]): Dictionary with chi-squared values
        alpha (float): Significance level controlling chi-squared threshold

    Returns:
        dict[str, float] | None: Dictionary with significant tokens.
        In case of corrupt input arguments, None is returned.
    """
