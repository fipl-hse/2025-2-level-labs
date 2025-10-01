"""
Lab 1
Extract keywords based on frequency related metrics
"""

import math

# pylint:disable=unused-argument
from typing import Any

def check_list(user_input: Any, elements_type: type, can_be_empty: bool) -> bool:
    if not isinstance(user_input, list):
        return False

    if not user_input:
        return can_be_empty


    for element in user_input:
        if not isinstance(element, elements_type):
            return False

    return True


def check_dict(user_input: Any, key_type: type, value_type: type, can_be_empty: bool) -> bool:
    if not isinstance(user_input, dict):
        return False

    if not user_input and not can_be_empty:
        return False

    for key, value in user_input.items():
        if not isinstance(key, key_type) or not isinstance(value, value_type):
            return False

    return True


def check_positive_int(user_input: Any) -> bool:
    if not isinstance(user_input, int) or isinstance(user_input, bool) or not user_input > 0:
        return False

    return True


def check_float(user_input: Any) -> bool:
    return isinstance(user_input, float)


def clean_and_tokenize(text: str) -> list[str] | None:
    if not isinstance(text, str):
        return None

    tokens = [''.join(symb.lower() for symb in word if symb.isalnum()) for word in text.split()]
    return [token for token in tokens if token]


def remove_stop_words(tokens: list[str], stop_words: list[str]) -> list[str] | None:
    if not check_list(tokens, str, True) or not check_list(stop_words, str, True):
        return None

    return [word for word in tokens if word not in stop_words]


def calculate_frequencies(tokens: list[str]) -> dict[str, int] | None:
    if not check_list(tokens, str, False):
        return None

    return {token: tokens.count(token) for token in tokens}


def get_top_n(frequencies: dict[str, int | float], top: int) -> list[str] | None:
    if not check_dict(frequencies, str,  int, False):
        if not check_dict(frequencies, str,  float, False):
            return None
    if not check_positive_int(top):
        return None
    sorted_freq = sorted(frequencies.keys(), key=lambda word: frequencies[word], reverse=True)
    top = min(len(frequencies), top)
    return sorted_freq[:top]


def calculate_tf(frequencies: dict[str, int]) -> dict[str, float] | None:
    if not check_dict(frequencies, str, int, False):
        return None

    words_in_text = sum(frequencies.values())
    tf_dict = {}
    for key, value in frequencies.items():
        tf_dict[key] = value / words_in_text

    return tf_dict


def calculate_tfidf(term_freq: dict[str, float], idf: dict[str, float]) -> dict[str, float] | None:
    if not check_dict(term_freq, str, float, False) or not check_dict(idf, str, float, True):
        return None

    for key, value in term_freq.items():
        idf_value = idf.get(key, math.log(47))
        term_freq[key] = value * idf_value
    return term_freq


def calculate_expected_frequency(doc_freqs: dict[str, int], corpus_freqs: dict[str, int]
) -> dict[str, float] | None:
    if not check_dict(doc_freqs, str, int, False) or not check_dict(corpus_freqs, str, int, True):
        return None
    expexted_freq_dict = {}
    words_in_doc = sum(doc_freqs.values())
    words_in_corpus = sum(corpus_freqs.values())

    for word in doc_freqs.keys():
        i = doc_freqs.get(word, 0)
        k = corpus_freqs.get(word, 0)
        l = words_in_doc - i
        m = words_in_corpus - k
        expexted_freq = ((i + k) * (i + l)) / (i + k + l + m)
        expexted_freq_dict[word] = expexted_freq
    return expexted_freq_dict


def calculate_chi_values(
    expected: dict[str, float], observed: dict[str, int]
) -> dict[str, float] | None:
    if not check_dict(expected, str, float, False) or not check_dict(observed, str, int, False):
        return None
    chi_values = {}
    for word, value in expected.items():
        exp_fr = value or 1
        obs_fr = observed.get(word, 0)
        chi = pow(obs_fr - exp_fr, 2) / exp_fr
        chi_values[word] = chi
    return chi_values


def extract_significant_words(chi_values: dict[str, float], alpha: float
) -> dict[str, float] | None:
    if not check_dict(chi_values, str, float, False) or not check_float(alpha):
        return None
    criterion = {0.05: 3.842, 0.01: 6.635, 0.001: 10.828}
    if alpha not in criterion:
        return None
    threshold = criterion[alpha]
    return {
        word: chi_values[word] for word, chi_value in chi_values.items()
        if chi_values[word] > threshold
        }