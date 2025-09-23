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
    else:
        return True
    if len(user_input)==0:
        return can_be_empty
    for el in user_input:
        if not isinstance(el, elements_type):
            return False

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
    if len(user_input)==0:
        return can_be_empty
    for k, v in user_input:
            return (isinstance(k, key_type), isinstance(v, value_type))


def check_positive_int(user_input: Any) -> bool:
    """
    Check if the object is a positive integer (not bool).

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """
    if (user_input,int and user_input>0):
        return True 
    else: False

def check_float(user_input: Any) -> bool:
    """
    Check if the object is a float.

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """
    if isinstance(user_input,float):
        return True
    else: 
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
    text=text.lower()
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for el in text:
        if el in punctuation:
            text=text.replace(el, "")
    text=text.split(' ')
    return [el for el in text if el.strip()]

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
    if not check_list(tokens, str, True) or not check_list(stop_words, str, True):
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
    if not isinstance(tokens, list):
        return None
    for token in tokens:
        if not isinstance(token, str):
            return None
    freq_tokens={}
    for token in tokens:
        freq_tokens [token]= tokens.count(token)
    return freq_tokens

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
    if not isinstance(frequencies, dict):
        return None 
    if not isinstance(top, int) or top<=0 or isinstance(top, bool):
        return None
    for k, v in frequencies.items():
        if not isinstance(k, str):
            return None 
        if not isinstance(v, int or float):
            return None 
    if len(frequencies)==0:
        return None
    sorted_frequencies = sorted(frequencies.items(), key = lambda x: x[1], reverse=True)
    top_el= sorted_frequencies[:top]
    if len(sorted_frequencies)<top:
        return [el[0] for el in sorted_frequencies]
    return [el[0] for el in top_el]

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
    for k, v in frequencies.items():
        if not isinstance(k, str):
            return None 
        if not isinstance(v, int):
            return None 
    N = sum(frequencies.values())
    if N==0:
        return {}
    tf_tokens={}
    for k,v in frequencies.items():
        tf_tokens[k] = v / N
    srt_tf=sorted(tf_tokens.items())
    srt_tf_tokens=dict(srt_tf)
    return srt_tf_tokens

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
    if not isinstance(term_freq, dict) or not isinstance(idf, dict) or len(term_freq)==0:
        return None
    for k, v in term_freq.items():
        if not isinstance(k, str):
            return None 
        if not isinstance(v, float):
            return None 
    for k, v in idf.items():
        if not isinstance(k, str):
            return None 
        if not isinstance(v, float):
            return None 
    tf_idf={}
    for k, v in term_freq.items():
        if k in idf:
            tf_idf[k] = v*idf[k]
        else:
            tf_idf[k] = v*math.log(47)
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
    if not isinstance(doc_freqs, dict) or not isinstance(corpus_freqs, dict)==0:
        return None
    for k, v in doc_freqs.items():
        if not isinstance(k, str):
            return None 
        if not isinstance(v, int):
            return None 
    for k, v in corpus_freqs.items():
        if not isinstance(k, str):
            return None 
        if not isinstance(v, int):
            return None 
    exp_freq_dict={}
    sum_in_doc = sum(doc_freqs.values())
    sum_in_corpus = sum(corpus_freqs.values())
    exp_freq_dict={}
    for t, j in doc_freqs.items():
        k=corpus_freqs.get(t, 0)
        l=sum_in_doc-j
        m=sum_in_corpus-k
        multi=(j+k)*(j+l)
        sum=j+k+l+m
        if sum==0:
            exp_freq_dict[t] = 0.0
        else:
            exp_freq_dict[t]=float(multi/sum)
        exp_freq_list=sorted(exp_freq_dict.items())
        exp_freq_dict=dict(exp_freq_list)
    return exp_freq_dict

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
