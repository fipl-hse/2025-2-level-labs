import math
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
    if can_be_empty:
        if user_input == []:
            return True
    else:
        if user_input == []:
            return False
    return bool(all(isinstance(step1, elements_type) for step1 in user_input))

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
    if can_be_empty:
        if user_input == {}:
            return True
    else:
        if user_input == {}:
            return False
    if all(isinstance(step1, key_type) for step1 in user_input.keys()):
        return bool(all(isinstance(step2, value_type) for step2 in user_input.values()))
    return False
def check_positive_int(user_input: Any) -> bool:
    """
    Check if the object is a positive integer (not bool).

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """
    return bool(isinstance(user_input, int) and user_input > 0)


def check_float(user_input: Any) -> bool:
    """
    Check if the object is a float.

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """
    return bool(isinstance(user_input, float))


def clean_and_tokenize(text: str) -> list[str] | None:
    """
    Remove punctuation, convert to lowercase, and split into tokens.

    Args:
        text (str): Original text

    Returns:
        list[str] | None: A list of lowercase tokens without punctuation.
        In case of corrupt input arguments, None is returned.
    """
    if isinstance(text, str):
        cleaned_text = []
        for i in text.split(' '):
            new_word = ''
            for j in i:
                if j.isalpha() or j == "'" or j.isdigit():
                    new_word += j.lower()
            if new_word != '':
                cleaned_text.append(new_word)
        return cleaned_text
    else:
        return None
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
    no_stop_words_text = []
    if isinstance(tokens, list) and isinstance(stop_words, list):
        if all(isinstance(step1, str) for step1 in tokens):
            if all(isinstance(step2, str) for step2 in stop_words):
                for i in tokens:
                    if i not in stop_words:
                        no_stop_words_text.append(i)
                if no_stop_words_text:
                    return no_stop_words_text
                else:
                    return []
            else:
                return None
def calculate_frequencies(tokens: list[str]) -> dict[str, int] | None:
    """
    Create a frequency dictionary from the token sequence.

    Args:
        tokens (list[str]): Token sequence

    Returns:
        dict[str, int] | None: A dictionary {token: occurrences}.
        In case of corrupt input arguments, None is returned.
    """
    dict_frequency = {}
    if isinstance(tokens, list) and all(isinstance(step1, str) for step1 in tokens):
        for i in tokens:
            dict_frequency.update({i : tokens.count(i)})
        return dict_frequency
    else:
        return None



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
    if isinstance(frequencies, dict) and isinstance(top, int):
        if frequencies == {}:
            return None
        if top <= 0:
            return None
        if top > len(frequencies):
            top = len(frequencies)
        top_frequencies = []
        if not isinstance(top, bool):
            if len(list(set(list(frequencies.values())))) == 1:
                top_frequencies = sorted(frequencies.items())
            else:
                top_frequencies = sorted(frequencies.items(), reverse=True)
            top_n_in_frequencies = []
            for i in range(top):
                if i < len(top_frequencies):
                    the_item = top_frequencies[i]
                    top_n_in_frequencies.append(the_item[0])
            return top_n_in_frequencies
        else:
            return None
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
    if not all(isinstance(step1, int) for step1 in frequencies.values()):
        return None
    if not all(isinstance(step2, str) for step2 in frequencies.keys()):
        return None
    new_dict = {}
    total_words = 0
    for word in frequencies.values():
        total_words += word
    for i, j in frequencies.items():
        new_tf = j/total_words
        new_dict.update({i : round(new_tf, 4)})
    return new_dict

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
    # if not check_dict(idf, str, float, False):
    #     return None
    tfidf_dict = {}
    if term_freq == {}:
        return None
    if idf == {}:
        for i, val in term_freq.items():
            tfidf_dict[i] = val * math.log(47 / 1)
        return tfidf_dict
    for i, val in term_freq.items():
        if i in idf:
            tfidf_dict[i] = val * idf[i]
        else:
            tfidf_dict[i] = val * math.log(47 / 1)
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
    if not check_dict(doc_freqs, str, int, False):
        return None
    if not check_dict(corpus_freqs, str, int, True):
        return None
    if corpus_freqs == {}:
        return doc_freqs
    expec_freqs = {}
    for i, val in doc_freqs.items():
        if i in corpus_freqs:
            expec_freqs[i] = (val + corpus_freqs[i])/5
        else:
            expec_freqs[i] = val/5
    return expec_freqs
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
    if not check_dict(expected, str, float, False):
        return None
    if not check_dict(observed, str, int, False):
        return None
    chi_freqs = {}
    for i, val in expected.items():
        if i in observed:
            chi_freqs[i] = (observed[i] - val)**2/val
        else:
            chi_freqs[i] = val
    return chi_freqs

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
    new_chi_dict = {}
    if not check_dict(chi_values, str, float, False):
        return None
    if not isinstance(alpha, float):
        return None
    for i in chi_values.items():
        if alpha in criterion.keys():
            crit = criterion.get(alpha)
            if crit != None:
                if i[1] > crit:
                    new_chi_dict.update({i})
        else:
            return None
    return new_chi_dict