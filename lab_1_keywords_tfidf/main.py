"""
Lab 1

Extract keywords based on frequency related metrics
"""


import math
from typing import Any

# pylint:disable=unused-argument



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
    return all(isinstance(element, elements_type) for element in user_input)



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
    if len(user_input) == 0:
        return can_be_empty
    return (all(isinstance(key, key_type) for key in user_input) and
        all(isinstance(value, value_type) for value in user_input.values()))


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
    if isinstance(user_input, float):
        return True
    return False


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
    cleaned_text = ''.join(symbol for symbol in text.lower()
                           if symbol.isalnum() or symbol.isspace())
    return cleaned_text.split()


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
    if not all([check_list(tokens, str, True),
        check_list(stop_words, str, True)]):
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
    if not check_list(tokens, str, True):
        return None
    frequencies = {}
    for token in tokens:
        if token in frequencies:
            frequencies[token] += 1
        else:
            frequencies[token] = 1
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
    if not all([isinstance(frequencies, dict), check_positive_int(top), frequencies]):
        return None
    for key, value in frequencies.items():
        if not isinstance(value, (int, float)) or not isinstance(key, str):
            return None
    top_n_tokens = [item[0] for item in sorted(frequencies.items(),
                                       key=lambda item: item[1], reverse=True)[:top]]
    return top_n_tokens


def calculate_tf(frequencies: dict[str, int]) -> dict[str, float] | None:
    """
    Calculate Term Frequency (TF) for each token.

    Args:
        frequencies (dict[str, int]): Raw occurrences of tokens

    Returns:
        dict[str, float] | None: Dictionary with tokens and TF values.
        In case of corrupt input arguments, None is returned.
    """
    if (not check_dict(frequencies, str, int, False) or
        sum(frequencies.values()) == 0 or not frequencies):
        return None
    tf_dict = {}
    dict_length = sum(frequencies.values())
    for token, nt in frequencies.items():
        tf_dict[token] = nt / dict_length
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
    if not all([check_dict(term_freq, str, float, False),
        check_dict(idf, str, float, True)]):
        return None
    tfidf_dict = {}
    for term, value in term_freq.items():
        tfidf_dict[term] = value * idf.get(term, math.log(47 / 1))
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
    if not all([check_dict(doc_freqs, str, int, False),
        check_dict(corpus_freqs, str, int, True)]):
        return None
    expected_frequency = {}
    total_doc = sum(doc_freqs.values())
    total_corpus = sum(corpus_freqs.values())
    for word in doc_freqs:
        word_in_doc = doc_freqs[word]
        word_in_corpus = corpus_freqs.get(word, 0)
        without_word_doc = total_doc - word_in_doc
        without_word_corpus = total_corpus - word_in_corpus
        formula = (((word_in_doc + word_in_corpus) *
                    (word_in_doc + without_word_doc)) /
                   (word_in_doc + word_in_corpus +
                    without_word_doc + without_word_corpus))
        expected_frequency[word] = formula
    return dict(sorted(expected_frequency.items()))


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
    if (not check_dict(expected, str, float, False) or
        not check_dict(observed, str, int, False)):
        return None
    chi_values = {}
    for term in observed:
        chi_values[term] = (((observed[term] - expected[term]) ** 2) /
                            expected[term])
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
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if (not check_dict(chi_values, str, float, False) or
        alpha not in criterion):
        return None
    significant_tokens = {token: value for token, value in chi_values.items()
                          if chi_values[token] > criterion[alpha]}
    return significant_tokens
