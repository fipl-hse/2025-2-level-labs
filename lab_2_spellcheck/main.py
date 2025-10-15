"""
Lab 2.
"""

# pylint:disable=unused-argument
from typing import Literal

from lab_1_keywords_tfidf.main import (
    check_dict,
    check_list,
)


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
    if not check_list(tokens,str,False):
        return None
    out_dict = {}
    for token in tokens:
        frequency = tokens.count(token)
        value_dict = frequency/len(tokens)
        out_dict[token] = value_dict
    return out_dict


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
    if not check_list(tokens,str,False):
        return None
    if not check_dict(vocabulary,str,float,False):
        return None
    alien_tokens = []
    for token in tokens:
        if token not in vocabulary:
            alien_tokens.append(token)
    return alien_tokens


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
    if not isinstance(token,str) or not isinstance(candidate,str):
        return None
    if not token or not candidate:
        return 1.0
    letters_1 = set(token)
    letters_2 = set(candidate)
    cross = letters_1.intersection(letters_2)
    unite = letters_1.union(letters_2)
    jaccard = 1-(len(cross)/len(unite))
    return jaccard


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
    if not first_token or not isinstance(first_token,str):
        return None
    if not check_dict(vocabulary,str,float,False):
        return None
    if (not isinstance(method,str) or
        method not in ["jaccard","frequency-based","levenshtein","jaro-winkler"]):
        return None
    if method == 'frequency-based':
        if alphabet is None or not check_list(alphabet,str,True):
            return {word: 1.0 for word in vocabulary}
        frequency_dist = calculate_frequency_distance(first_token,vocabulary,alphabet)
        return frequency_dist
    sum_distance={}
    for key in vocabulary:
        token_distance = None
        if method == 'jaccard':
            token_distance = calculate_jaccard_distance(first_token,key)
        elif method == 'levenshtein':
            token_distance = calculate_levenshtein_distance(first_token,key)
        if token_distance is None:
            return None
        sum_distance[key] = token_distance
    return sum_distance


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
    if alphabet is None:
        alphabet = []
    if not isinstance(wrong_word,str):
        return None
    if not check_dict(vocabulary,str,float,False):
        return None
    if not isinstance(method,str):
        return None
    if method not in ["jaccard","frequency-based","levenshtein","jaro-winkler"]:
        return None
    if not check_list(alphabet,str,True):
        return None
    best_variants = []
    variants = calculate_distance(wrong_word,vocabulary,method,alphabet)
    if variants is None:
        return None
    min_dist = min(variants.values())
    for key,value in variants.items():
        if value == min_dist:
            best_variants.append(key)
    if not best_variants:
        return None
    if len(best_variants) == 1:
        return best_variants[0]
    best_variants.sort(key = lambda word: (abs(len(word) - len(wrong_word)), word))
    return best_variants[0]


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
    if not isinstance(token_length,int) or token_length < 0:
        return None
    if not isinstance(candidate_length,int) or candidate_length < 0:
        return None
    matrix=[[0] * (candidate_length + 1) for _ in range(token_length + 1)]
    for i in range (candidate_length + 1):
        matrix[0][i] = i
    for j in range (token_length + 1):
        matrix[j][0] = j
    return matrix


def fill_levenshtein_matrix(token: str, candidate: str) -> list[list[int]] | None:
    """
    Fill a Levenshtein matrix with edit distances between all prefixes.

    Args:
        token (str): First string.
        candidate (str): Second string.

    Returns:
        list[list[int]] | None: Completed Levenshtein distance matrix.
    """
    if not isinstance(token,str) or not isinstance(candidate,str):
        return None
    matrix = initialize_levenshtein_matrix(len(token),len(candidate))
    if matrix is None:
        return None
    for i in range (1, len(token) + 1):
        for j in range (1, len(candidate) + 1):
            if token[i - 1] == candidate[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost
            )
    return matrix


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
    if not isinstance(token,str) or len(token) < 0:
        return None
    if not isinstance(candidate,str) or len(candidate) < 0:
        return None
    matrix=fill_levenshtein_matrix(token,candidate)
    if matrix is None:
        return None
    return matrix[-1][-1]


def delete_letter(word: str) -> list[str]:
    """
    Generate all possible words by deleting one letter from the word.

    Args:
        word (str): The input incorrect word.

    Returns:
        list[str]: A sorted list of words with one letter removed at each position.

    In case of corrupt input arguments, empty list is returned.
    """
    if not isinstance(word,str):
        return []
    candidates = []
    for i in range(len(word)):
        new_word = word[:i] + word[i+1:]
        candidates.append(new_word)
    return sorted(candidates)


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
    if not isinstance(word,str):
        return []
    if not check_list(alphabet,str,False):
        return []
    candidates = []
    for i in range(len(word) + 1):
        for letter in alphabet:
            new_word = word[:i] + letter + word[i:]
            candidates.append(new_word)
    return sorted(candidates)


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
    if not isinstance(word,str):
        return []
    if not check_list(alphabet,str,False):
        return []
    replacements = []
    for i,value in enumerate(word):
        for letter in alphabet:
            if letter != value:
                new_word = word[:i] + letter + word[i+1:]
                replacements.append(new_word)
    return sorted(replacements)


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
    if not isinstance(word,str):
        return []
    swapping = []
    for i in range(len(word) - 1):
        new_word=word[:i] + word[i+1] + word[i] + word[i+2:]
        swapping.append(new_word)
    return sorted(swapping)


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
    if not isinstance(word,str):
        return None
    if not check_list(alphabet,str,True):
        return None
    candidates = []
    candidates.extend(delete_letter(word))
    candidates.extend(add_letter(word,alphabet))
    candidates.extend(replace_letter(word,alphabet))
    candidates.extend(swap_adjacent(word))
    return sorted(set(candidates))


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
    if not isinstance(word,str):
        return None
    if not check_list(alphabet,str,True):
        return None
    candidates = set()
    first_candidates = generate_candidates(word,alphabet)
    if first_candidates is None:
        return None
    candidates.update(first_candidates)
    for token in first_candidates:
        second_candidates = generate_candidates(token,alphabet)
        if second_candidates is None:
            return None
        candidates.update(second_candidates)
    return tuple(sorted(candidates))


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
    if not isinstance(word,str):
        return None
    if not frequencies or not isinstance(frequencies,dict):
        return None
    if not check_list(alphabet,str,True):
        return None
    for token,value in frequencies.items():
        if not isinstance(token,str) or not isinstance(value,(int,float)):
            return None
    candidates = propose_candidates(word,alphabet)
    result: dict[str,float] = {}
    for token in frequencies:
        result[token] = 1.0
    if not candidates or candidates is None:
        return result
    for version in candidates:
        if version in frequencies:
            result[version] = 1 - frequencies[version]
    return result


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
