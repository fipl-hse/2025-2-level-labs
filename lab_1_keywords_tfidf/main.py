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
    
    if not can_be_empty and len(user_input) == 0:
        return False
    
    for key, value in user_input.items():
        if not isinstance(key, key_type) or not isinstance(value, value_type):
            return False


def check_positive_int(user_input: Any) -> bool:
    """
    Check if the object is a positive integer (not bool).

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(user_input, int) and user_input > 0


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
    import string
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if len(stripped) == 0:
        return []
    punctuation_chars = string.punctuation
    translation_table = str.maketrans({ch: None for ch in punctuation_chars})
    no_punct = stripped.translate(translation_table)
    tokens = no_punct.lower().split()

    return tokens
    



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
    if tokens is None or stop_words is None:
        return None
    if not isinstance(tokens, list) or not isinstance(stop_words, list):
        return None

    stop_set = set(s.lower() for s in stop_words)

    result = [t for t in tokens if t.lower() not in stop_set]

    return result


def calculate_frequencies(tokens: list[str]) -> dict[str, int] | None:
    """
    Create a frequency dictionary from the token sequence.

    Args:
        tokens (list[str]): Token sequence

    Returns:
        dict[str, int] | None: A dictionary {token: occurrences}.
        In case of corrupt input arguments, None is returned.
    """
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        return None

    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return freq


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
    if not isinstance(top, int) or top < 0:
        return None
    
    # Validate contents of frequencies
    for token, freq in frequencies.items():
        if not isinstance(token, str):
            return None
        if not isinstance(freq, (int, float)) or freq < 0:
            return None

    # --- Main Logic ---
    # Handle edge case: if top is 0, return an empty list.
    if top == 0:
        return []
    
    # Create a list of (token, frequency) tuples to sort.
    sorted_tokens_with_freq = sorted(
        frequencies.items(),
        key=lambda item: item[1],  # Sort by frequency (the second element of the tuple)
        reverse=True  # Descending order
    )

    # Extract the top 'top' tokens from the sorted list.
    top_n_items = sorted_tokens_with_freq[:top]

    # Extract just the tokens from the top N items.
    top_n_tokens = [token for token, freq in top_n_items]

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
    if not isinstance(frequencies, dict):
        return None
    if not frequencies:
        return {}

    total_tokens = 0
    for token, count in frequencies.items():
        if not isinstance(token, str) or not isinstance(count, int) or count < 0:
            return None
        total_tokens += count

    if total_tokens == 0:
        return {token: 0.0 for token in frequencies} 

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
    if not isinstance(term_freq, dict) or not isinstance(idf, dict):
        return None

    tfidf_scores = {}
    for token, tf_score in term_freq.items():
        if not isinstance(token, str) or not isinstance(tf_score, (int, float)) or tf_score < 0:
            return None

        if token in idf:
            idf_score = idf[token]
            if not isinstance(idf_score, (int, float)) or idf_score < 0:
                return None
            tfidf_scores[token] = tf_score * idf_score
        else:
            tfidf_scores[token] = 0.0

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
    if not isinstance(doc_freqs, dict) or not isinstance(corpus_freqs, dict):
        return None

    total_doc_tokens = 0
    for token, count in doc_freqs.items():
        if not isinstance(token, str) or not isinstance(count, int) or count < 0:
            return None
        total_doc_tokens += count

    total_corpus_tokens = 0
    for token, count in corpus_freqs.items():
        if not isinstance(token, str) or not isinstance(count, int) or count < 0:
            return None
        total_corpus_tokens += count

    if total_doc_tokens == 0:
        return {token: 0.0 for token in doc_freqs}

    if total_corpus_tokens == 0:
        return None 

    expected_frequencies = {}
    for token, doc_count in doc_freqs.items():
        corpus_count = corpus_freqs.get(token, 0) 
        expected_freq = (total_doc_tokens / total_corpus_tokens) * corpus_count
        expected_frequencies[token] = expected_freq

    return expected_frequencies


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

    chi_squared_values = {}

    for token, obs_freq in observed.items():
        if not isinstance(token, str) or not isinstance(obs_freq, int) or obs_freq < 0:
            return None

        if token in expected:
            exp_freq = expected[token]

            if not isinstance(exp_freq, (int, float)) or exp_freq < 0:
                return None

            if exp_freq == 0:
                if obs_freq == 0:
                    chi_squared_values[token] = 0.0
                
            else:
                chi_squared_values[token] = ((obs_freq - exp_freq) ** 2) / exp_freq
    return chi_squared_values
    


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
    if not isinstance(chi_values, dict):
        return None
    if not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
        return None

    for token, chi_val in chi_values.items():
        if not isinstance(token, str):
            return None
        if not isinstance(chi_val, (int, float)):
            return None
        if chi_val < 0 and chi_val != float('-inf'): 
            return None
    critical_thresholds = {
        0.05: 3.841, 
        0.01: 6.635,  
        0.001: 10.828 
    }

    critical_value = None
    epsilon = 1e-9
    for key_alpha, val in critical_thresholds.items():
        if abs(alpha - key_alpha) < epsilon:
            critical_value = val
            break
    if critical_value is None:
        return None

    significant_words = {}
    for token, chi_val in chi_values.items():
        if chi_val == float('inf'): 
            significant_words[token] = chi_val
        elif chi_val > critical_value:
            significant_words[token] = chi_val

    return significant_words

