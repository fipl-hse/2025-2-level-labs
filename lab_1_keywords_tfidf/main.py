"""
Lab 1

Extract keywords based on frequency related metrics
"""

import math

# pylint:disable=unused-argument
from typing import Any


def check_list(user_input: Any, elements_type: type, can_be_empty: bool) -> bool:
    '''
    Check if the object is a list containing elements of a certain type.
    Args:
        user_input (Any): Object to check
        elements_type (type): Expected type of list elements
        can_be_empty (bool): Whether an empty list is allowed

    Returns:
        bool: True if valid, False otherwise
    '''

    if not isinstance(user_input, list):
        return False
    if not can_be_empty and len(user_input)==0:
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
    if not can_be_empty and len(user_input)==0:
        return False
    for key in user_input:
        if not isinstance(key, key_type):
            return False
    for value in user_input.values():
        if not isinstance(value, value_type):
            return False
    return True

def check_positive_int(user_input: Any) -> bool:
    """
    Check if the object is a positive integer (not bool).
    """
    return isinstance(user_input, int) and not isinstance(user_input, bool)

def check_float(user_input: Any) -> bool:

    """
    Check if the object is a float.
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

    punctuation='!@#$%^&*(\')"№;:?,./<>`{~}[]+=|№-_'

    new_text=''
    for element in text:
        if element not in punctuation:
            new_text+=element

    low_register=new_text.lower()

    splited_text=low_register.split()

    return splited_text



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

    if not check_list(tokens, str, True):
        return None
    if not check_list(stop_words, str, True):
        return None

    text_without_stop_words=[]

    for element in tokens:
        if element not in stop_words:
            text_without_stop_words.append(element)

    return text_without_stop_words

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

    dictionary = {}

    for element in tokens:
        if not isinstance(element, str):
            return None
        if element in dictionary:
            dictionary[element] += 1
        else:
            dictionary[element] = 1

    return dictionary

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
    if not check_positive_int(top) or top<=0:
        return None

    if not (check_dict(frequencies, str, int, False) or check_dict(frequencies, str, float, False)):
        return None

    sorted_dictionary=sorted(frequencies.items(), key=lambda x: x[1], reverse=True)

    top_n_words=[]
    for word, _ in sorted_dictionary[:top]:
        top_n_words.append(word)
    return top_n_words

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

    quantity_of_words=sum(frequencies.values())

    dictionary={}
    for word, value in frequencies.items():
        calculated_frequency=value/quantity_of_words
        dictionary[word]=calculated_frequency
    return dictionary

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

    idf_without_entering=math.log(47/1)

    dictionary_tfidf={}
    for token, value in term_freq.items():
        if token in idf:
            dictionary_tfidf_value=idf[token]
        else:
            dictionary_tfidf_value=idf_without_entering

        dictionary_tfidf[token]=value*dictionary_tfidf_value

    return dictionary_tfidf

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
