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
    if not can_be_empty and len(user_input) == 0:
        return False
    return all(isinstance(el, elements_type) for el in user_input)


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
    if not can_be_empty and len(user_input) == 0:
        return False
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
    text = text.strip().lower()
    tokens = [char for char in text if char.isalpha()]
    return list("".join(tokens))


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
    if not isinstance(tokens, list) or not isinstance(stop_words, list):
        return None
    
    if not all(isinstance(t, str) for t in tokens):
        return None
    if not all(isinstance(s, str) for s in stop_words):
        return None
    
    filtrovany_tokens = [t for t in tokens if t not in stop_words]
    return filtrovany_tokens


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
    if not all (isinstance(t,str) for t in tokens):
        return None
    
    frequences = {}
    for t in tokens:
        if t in frequences:
            frequences[t] += 1
        else:
            frequences[t] = 1
    return frequences



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
    if not isinstance(top, int) or top <= 0:
        return None
    

    if len(frequencies) == 0:
        return []
    
    items_sorted = sorted(frequencies.items(), key=lambda kv: (-kv[1], kv[0]))

    top_n = [token for token, _ in items_sorted[:top]]
    return top_n


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
    if not all(isinstance(k,str) and isinstance(v,int) for k, v in frequencies.items()):
        return None
    
    total_count = sum(frequencies.values())
    if total_count == 0:
        return {}
    
    tf_dict = {k: v / total_count for k, v in frequencies.items()}
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
    if not isinstance(term_freq, dict) or not isinstance(idf, dict):
        return None
    if not all(isinstance(k, str) and isinstance(v, float) for k, v in term_freq.items()):
        return None
    if not all(isinstance(k, str) and isinstance(v, float) for k, v in idf.items()):
        return None
    
    tdidf = {token: term_freq[token] * idf.get(token, 0) for token in term_freq}
    return tdidf


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
    if not isinstance(doc_freqs, dict) or not isinstance(corpus_freqs, dict):
        return None
    if not all(isinstance(k, str) and isinstance(v, int) for k, v in doc_freqs.items()):
        return None
    if not all(isinstance(k,str) and isinstance(v, int) for k, v in corpus_freqs.items()):
        return None
    
    total_doc = sum(doc_freqs.values())
    total_corpus = sum(corpus_freqs.values())
    if total_doc == 0 or total_corpus == 0:
        return {}
    
    expected = {k: (doc_freqs.get(k,0) * corpus_freqs.get(k, 0) / total_corpus) for k in doc_freqs}
    return expected


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
    if not isinstance(expected, dict) or not isinstance(observed, dict):
        return None
    
    chi = {}
    for token in observed:
        exp = expected.get(token, 0)
        obs = observed.get(token)
        if exp > 0:
            chi[token] = ((obs - exp) ** 2) / exp
        else:
            chi[token] = 0
    return chi


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
    if not isinstance(chi_values, dict) or not isinstance(alpha,float):
        return None
    if not all(isinstance(k, str) and isinstance(v, float) for k, v in chi_values.items()):
        return None
    if alpha<=0:
        return {}
    
    significant = {token: chi for token, chi in chi_values.items() if chi > alpha}
    return significant
    

if __name__ == "__main__":
    with open(r"C:\Users\PC_TEMUR\Desktop\Github Teimur Project\2025-2-level-labs\lab_1_keywords_tfidf\Дюймовочка.txt", encoding="utf-8") as f:
        corpus_text = f.read()
    stop_words = ["и","в","на"]
    
    corpus_tokens = clean_and_tokenize(corpus_text)
    corpus_tokens_filtrovany = remove_stop_words(corpus_tokens, stop_words)
    corpus_freqs = calculate_frequencies(corpus_tokens_filtrovany)

    moi_text = corpus_text
    tokens = clean_and_tokenize(moi_text)
    filtrovany = remove_stop_words(tokens, stop_words)
    frequencies = calculate_frequencies(filtrovany)

    top1 = get_top_n(frequencies, 1)
    print('Top1:', top1)

    top3 = get_top_n(frequencies, 3)
    print('Top3', top3)

    tf_dict = calculate_tf(frequencies)
    print('TF:', tf_dict)

    idf_forexample = {token:1.0 for token in tf_dict}
    tfidf_scores = calculate_tfidf(tf_dict, idf_forexample)
    print("TF-IDF", tfidf_scores)

    top_tfidf = get_top_n(tfidf_scores,10)
    print("TOP-10 TF-IDF Tokens", top_tfidf)

    expected_freqs = calculate_expected_frequency(frequencies, corpus_freqs)
    print("Expected Freqs", expected_freqs)

    chi_values = calculate_chi_values(expected_freqs, frequencies)
    print("Chi values", chi_values)

    alpha = 2.0
    significant_words = extract_significant_words(chi_values, alpha)
    print("Significant tokens", list(significant_words.keys()))

    top_significant = get_top_n(significant_words, 10)
    print("TOP-10 Significant tokens", top_significant)