"""
Lab 2.
"""

# pylint:disable=unused-argument
from typing import Literal, Dict, Optional, List


def build_vocabulary(tokens: list[str]) -> dict[str, float] | None:
    """
    Build a vocabulary from the documents.

    Args:
        tokens (list[str]): List of tokens.

    Returns:
        dict[str, float] | None: Dictionary with words and relative
        frequencies as keys and values respectively.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(tokens, list):
        return None
    if any(not isinstance(token, str) for token in tokens):
        return None
    if tokens==[]:
        return None
    vocabulary = {}
    for token in tokens:
        if token in vocabulary:
            vocabulary[token] += 1
        else:
            vocabulary[token] = 1
    total_tokens = len(tokens)
    for token in vocabulary:
        vocabulary[token] = vocabulary[token] / total_tokens
    return vocabulary

def find_out_of_vocab_words(tokens: list[str], vocabulary: dict[str, float]) -> list[str] | None:
    """
    Found words out of vocabulary.

    Args:
        tokens (list[str]): List of tokens.
        vocabulary (dict[str, float]): Dictionary with unique words and their relative frequencies.

    Returns:
        list[str] | None: List of incorrect words.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(tokens, list) or not isinstance(vocabulary, dict):
        return None
    if not all(isinstance(token, str) for token in tokens):
        return None
    if not all(isinstance(word, str) and isinstance(freq, float) for word, freq in vocabulary.items()):
        return None
    out_of_vocab_words = [token for token in tokens if token not in vocabulary]
    return out_of_vocab_words

def calculate_jaccard_distance(token: str, candidate: str) -> float | None:
    """
    Calculate Jaccard distance between two strings.

    Args:
        token (str): First string to compare.
        candidate (str): Second string to compare.

    Returns:
        float | None: Jaccard distance score in range [0, 1].

    In case of corrupt input arguments, None is returned.
    In case of both strings being empty, 0.0 is returned.
    """
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None
    if token == "" and candidate == "":
        return 1.0  
    set1 = set(token)
    set2 = set(candidate)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 1.0
    jaccard_distance = 1.0 - (float(intersection) / union)
    return jaccard_distance

def calculate_frequency_based_distance(first_token: str, vocabulary: Dict[str, float]) -> Dict[str, float]:
    if first_token in vocabulary:
        return {"frequency-based": 0.0}
    else:
        return {"frequency-based": 1.0}
    
def calculate_levenshtein_distance(first_token: str, vocabulary: Dict[str, float], alphabet: Optional[List[str]] = None) -> Dict[str, float]:
    default_alphabet = list("abcdefghijklmnopqrstuvwxyz")  
    local_alphabet = alphabet if alphabet else default_alphabet
    
    def edit_distance(s1: str, s2: str) -> int:
        n, m = len(s1), len(s2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[n][m]
    min_distance = float('inf')
    for word in vocabulary:
        distance = edit_distance(first_token, word)
        min_distance = min(min_distance, distance)
    return {"levenshtein": float(min_distance)}

def calculate_jaro_winkler_distance(first_token: str, vocabulary: Dict[str, float], alphabet: Optional[List[str]] = None) -> Dict[str, float]:
    def jaro_distance(s1: str, s2: str) -> float:
        len_s1, len_s2 = len(s1), len(s2)
        if len_s1 == 0 and len_s2 == 0:
            return 1.0
        match_distance = (max(len_s1, len_s2) // 2) - 1
        matches = 0
        transpositions = 0
        s1_matches = [False] * len_s1
        s2_matches = [False] * len_s2
        for i in range(len_s1):
            start = max(0, i - match_distance)
            end = min(len_s2, i + match_distance + 1)
            for j in range(start, end):
                if s1[i] == s2[j] and not s2_matches[j]:
                    s1_matches[i] = True
                    s2_matches[j] = True
                    matches += 1
                    break
        if matches == 0:
            return 0.0
        k = 0
        for i in range(len_s1):
            if s1_matches[i]:
                while not s2_matches[k]:
                    k += 1
                if s1[i] != s2[k]:
                    transpositions += 1
                k += 1
        return (1/3) * (matches / len_s1 + matches / len_s2 + (matches - transpositions / 2) / matches)

    def jaro_winkler_similarity(s1: str, s2: str) -> float:
        jaro_sim = jaro_distance(s1, s2)
        if jaro_sim > 0.7:  
            prefix_length = 0
            for i in range(min(len(s1), len(s2))):
                if s1[i] == s2[i]:
                    prefix_length += 1
                else:
                    break
            prefix_length = min(4, prefix_length) 
            jaro_winkler_sim = jaro_sim + (0.1 * prefix_length * (1 - jaro_sim))
        else:
            jaro_winkler_sim = jaro_sim
        return jaro_winkler_sim
    min_distance = float('inf')
    for word in vocabulary:
        distance = 1 - jaro_winkler_similarity(first_token, word) 
        min_distance = min(min_distance, distance)
    return {"jaro-winkler": float(min_distance)}

def calculate_distance(
    first_token: str,
    vocabulary: dict[str, float],
    method: Literal["jaccard", "frequency-based", "levenshtein", "jaro-winkler"],
    alphabet: list[str] | None = None,
) -> dict[str, float] | None:
    """
    Calculate distance between two strings using the specified method.

    Args:
        first_token (str): First string to compare.
        vocabulary (dict[str, float]): Dictionary mapping words to their relative frequencies.
        method (str): Method to use for comparison.
        alphabet (list[str]): The alphabet with letters.

    Returns:
        dict[str, float] | None: Calculated distance score.

    In case of corrupt input arguments or unsupported method, None is returned.
    """
    if not first_token or not vocabulary:
        return None
    if method == "jaccard":
        return calculate_jaccard_distance(first_token, vocabulary)
    elif method == "frequency-based":
        return calculate_frequency_based_distance(first_token, vocabulary)
    elif method == "levenshtein":
        return calculate_levenshtein_distance(first_token, vocabulary, alphabet)
    elif method == "jaro-winkler":
        return calculate_jaro_winkler_distance(first_token, vocabulary, alphabet)
    else:
        return None


def find_correct_word(
    wrong_word: str,
    vocabulary: dict[str, float],
    method: Literal["jaccard", "frequency-based", "levenshtein", "jaro-winkler"],
    alphabet: list[str] | None = None,
) -> str | None:
    """
    Find the most similar word from vocabulary using the specified method.

    Args:
        wrong_word (str): Word that might be misspelled.
        vocabulary (dict[str, float]): Dict of candidate words.
        method (str): Method to use for comparison.
        alphabet (list[str]): The alphabet with letters.

    Returns:
        str | None: Word from vocabulary with the lowest distance score.
             In case of ties, the closest in length and lexicographically first is chosen.

    In case of empty vocabulary, None is returned.
    """
    if not vocabulary:
        return None
    best_match: Optional[str] = None
    min_score: float = float('inf')
    best_length = float('inf')
    best_lex_order: str = "~"
    for candidate in vocabulary.keys():
        score = calculate_distance(wrong_word, candidate, method, alphabet)
        is_tie = (score == min_score)
        if score < min_score:
            min_score = score
            best_match = candidate
            best_length = len(candidate)
            best_lex_order = candidate
        elif is_tie:
            candidate_length = len(candidate)
            if candidate_length < best_length:
                best_match = candidate
                best_length = candidate_length
                best_lex_order = candidate  
            elif candidate_length == best_length:
                if candidate < best_lex_order:
                    best_match = candidate
                    best_lex_order = candidate
    return best_match


def initialize_levenshtein_matrix(
    token_length: int, candidate_length: int
) -> list[list[int]] | None:
    """
    Initialize a 2D matrix for Levenshtein distance calculation.

    Args:
        token_length (int): Length of the first string.
        candidate_length (int): Length of the second string.

    Returns:
        list[list[int]] | None: Initialized matrix with base cases filled.
    """


def fill_levenshtein_matrix(token: str, candidate: str) -> list[list[int]] | None:
    """
    Fill a Levenshtein matrix with edit distances between all prefixes.

    Args:
        token (str): First string.
        candidate (str): Second string.

    Returns:
        list[list[int]] | None: Completed Levenshtein distance matrix.
    """
    


def calculate_levenshtein_distance(token: str, candidate: str) -> int | None:
    """
    Calculate the Levenshtein edit distance between two strings.

    Args:
        token (str): First string.
        candidate (str): Second string.

    Returns:
        int | None: Minimum number of single-character edits (insertions, deletions,
             substitutions) required to transform token into candidate.
    """


def delete_letter(word: str) -> list[str]:
    """
    Generate all possible words by deleting one letter from the word.

    Args:
        word (str): The input incorrect word.

    Returns:
        list[str]: A sorted list of words with one letter removed at each position.

    In case of corrupt input arguments, empty list is returned.
    """


def add_letter(word: str, alphabet: list[str]) -> list[str]:
    """
    Generate all possible words by inserting a letter from the alphabet
    at every possible position in the word.

    Args:
        word (str): The input incorrect word.
        alphabet (list[str]): The alphabet with letters.

    Returns:
        list[str]: A list of words with one additional letter inserted.

    In case of corrupt input arguments, empty list is returned.
    """


def replace_letter(word: str, alphabet: list[str]) -> list[str]:
    """
    Generate all possible words by replacing each letter in the word
    with letters from the alphabet.

    Args:
        word (str): The input incorrect word.
        alphabet (list[str]): The alphabet with letters.

    Returns:
        list[str]: A sorted list of words with one letter replaced at each position.

    In case of corrupt input arguments, empty list is returned.
    """


def swap_adjacent(word: str) -> list[str]:
    """
    Generate all possible words by swapping each pair of adjacent letters
    in the word.

    Args:
        word (str): The input incorrect word.

    Returns:
        list[str]: A sorted list of words where two neighboring letters are swapped.

    In case of corrupt input arguments, empty list is returned.
    """


def generate_candidates(word: str, alphabet: list[str]) -> list[str] | None:
    """
    Generate all possible candidate words for a given word using
    four basic operations.

    Args:
        word (str): The input word.
        alphabet (list[str]): Alphabet for candidates creation.

    Returns:
        list[str] | None: A combined list of candidate words generated by all operations.

    In case of corrupt input arguments, None is returned.
    """


def propose_candidates(word: str, alphabet: list[str]) -> tuple[str, ...] | None:
    """
    Generate candidate words by applying single-edit operations
    (delete, add, replace, swap) to the word.

    Args:
        word (str): The input incorrect word.
        alphabet (list[str]): Alphabet for candidates creation.

    Returns:
        tuple[str] | None: A tuple of unique candidate words generated from the input.

    In case of corrupt input arguments, None is returned.
    """


def calculate_frequency_distance(
    word: str, frequencies: dict, alphabet: list[str]
) -> dict[str, float] | None:
    """
    Suggest the most probable correct spelling for the word.

    Args:
        word (str): The input incorrect word.
        frequencies (dict): A dictionary with frequencies.
        alphabet (list[str]): Alphabet for candidates creation.

    Returns:
        dict[str, float] | None: The most probable corrected word.

    In case of corrupt input arguments, None is returned.
    """


def get_matches(
    token: str, candidate: str, match_distance: int
) -> tuple[int, list[bool], list[bool]] | None:
    """
    Find matching letters between two strings within a distance.

    Args:
        token (str): The first string to compare.
        candidate (str): The second string to compare.
        match_distance (int): Maximum allowed offset for letters to be considered matching.

    Returns:
        tuple[int, list[bool], list[bool]]:
            Number of matching letters.
            Boolean list indicating matches in token.
            Boolean list indicating matches in candidate.

    In case of corrupt input arguments, None is returned.
    """


def count_transpositions(
    token: str, candidate: str, token_matches: list[bool], candidate_matches: list[bool]
) -> int | None:
    """
    Count the number of transpositions between two strings based on matching letters.

    Args:
        token (str): The first string to compare.
        candidate (str): The second string to compare.
        token_matches (list[bool]): Boolean list indicating matches in token.
        candidate_matches (list[bool]): Boolean list indicating matches in candidate.

    Returns:
        int | None: Number of transpositions.

    In case of corrupt input arguments, None is returned.
    """


def calculate_jaro_distance(
    token: str, candidate: str, matches: int, transpositions: int
) -> float | None:
    """
    Calculate the Jaro distance between two strings.

    Args:
        token (str): The first string to compare.
        candidate (str): The second string to compare.
        matches (int): Number of matching letters.
        transpositions (int): Number of transpositions.

    Returns:
        float | None: Jaro distance score.

    In case of corrupt input arguments, None is returned.
    """


def winkler_adjustment(
    token: str, candidate: str, jaro_distance: float, prefix_scaling: float = 0.1
) -> float | None:
    """
    Apply the Winkler adjustment to boost distance for strings with a common prefix.

    Args:
        token (str): The first string to compare.
        candidate (str): The second string to compare.
        jaro_distance (float): Jaro distance score.
        prefix_scaling (float): Scaling factor for the prefix boost.

    Returns:
        float | None: Winkler adjustment score.

    In case of corrupt input arguments, None is returned.
    """


def calculate_jaro_winkler_distance(
    token: str, candidate: str, prefix_scaling: float = 0.1
) -> float | None:
    """
    Calculate the Jaro-Winkler distance between two strings.

    Args:
        token (str): The first string.
        candidate (str): The second string.
        prefix_scaling (float): Scaling factor for the prefix boost.

    Returns:
        float | None: Jaro-Winkler distance score.

    In case of corrupt input arguments or corrupt outputs of used functions, None is returned.
    """
