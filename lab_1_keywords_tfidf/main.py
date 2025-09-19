"""
Lab 1

Extract keywords based on frequency related metrics
"""

# pylint:disable=unused-argument
import math
from typing import Any, Mapping, Union


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
    if not user_input:
        return can_be_empty
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
    cleaned_and_tokenized_text = []
    for word in text.split():
        cleaned_word = (''.join(symbol.lower()
                                        for symbol in word if symbol.isalnum()))
        if cleaned_word:
            cleaned_and_tokenized_text.append(cleaned_word)
    return cleaned_and_tokenized_text


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
    cleaned_tokens = [token for token in tokens if token not in stop_words]
    return cleaned_tokens


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
    freq_dict = {token: tokens.count(token) for token in tokens}
    return freq_dict


def get_top_n(frequencies: Mapping[str, Union[int, float]], top: int) -> list[str] | None:
    """
    Extract the most frequent tokens.

    Args:

        frequencies (dict[str, int | float]): A dictionary with tokens and their frequencies
        top (int): Number of tokens to extract

    Returns:
        list[str] | None: Top-N tokens sorted by frequency.
        In case of corrupt input arguments, None is returned.
    """
    if (not isinstance(frequencies, dict) or not frequencies
        or not check_positive_int(top)):
        return None
    for k, v in frequencies.items():
        if not isinstance(k, str) or not isinstance(v, (float, int)):
            return None
    word_lst_sorted = []
    freq_lst = list(frequencies.items())
    freq_lst_sorted = sorted(freq_lst, key = lambda x: (-x[-1], x[0]))
    top = min(top, len(freq_lst_sorted))
    word_lst_sorted = [item[0] for item in freq_lst_sorted[:top]]
    return word_lst_sorted


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
    tf_dict = {k: v/sum(frequencies.values()) for k, v in frequencies.items()}
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
    if not check_dict(term_freq, str, float, False) or not check_dict(idf, str, float, True):
        return None
    tf_idf_dict = {token: term_freq[token] * idf[token]
                  if token in idf else term_freq[token] * math.log(47) for token in term_freq}
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
    if not check_dict(doc_freqs, str, int, False) or not check_dict(corpus_freqs, str, int, True):
        return None
    expected_freq = {}
    doc_freqs_sum = sum(doc_freqs.values())
    corpus_freqs_sum = sum(corpus_freqs.values())
    for token, frequency in doc_freqs.items():
        t_count_in_doc = frequency
        without_t_count_in_doc = doc_freqs_sum - t_count_in_doc
        t_count_in_corpus = corpus_freqs.get(token, 0)
        without_t_count_in_corpus = corpus_freqs_sum - t_count_in_corpus
        expected_value = (t_count_in_doc +
                          t_count_in_corpus)*(t_count_in_doc + without_t_count_in_doc)
        expected_value = expected_value/(t_count_in_doc + t_count_in_corpus
                                         + without_t_count_in_doc + without_t_count_in_corpus)
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
    chi_values_dict = {}
    for k, v in expected.items():
        chi = (observed[k] - v)**2 / v
        chi_values_dict[k] = chi
    return chi_values_dict



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
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if alpha not in criterion:
        return None
    critical_value = criterion[alpha]
    significant_chi_values = {}
    for k, v in chi_values.items():
        if v > critical_value:
            significant_chi_values[k] = v
    return significant_chi_values
