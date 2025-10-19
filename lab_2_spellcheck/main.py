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
    if not check_list(tokens, str, False):
        return None
    vocabulary = {}
    for el in tokens:
        if el not in vocabulary.keys():
            vocabulary[el] = tokens.count(el)/len(tokens)
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
    if not check_list(tokens, str, False):
        return None
    if not check_dict(vocabulary, str, float, False):
        return None
    weird_tokens = []
    for el in tokens:
        if el not in vocabulary:
            weird_tokens.append(el)
    return weird_tokens


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
    if not isinstance(token, str):
        return None
    if not isinstance(candidate, str):
        return None
    if token == '' or candidate == '':
        return 1.0
    token_set = set(token)
    candidate_set = set(candidate)
    jaccard_distance = 1 - len(token_set.intersection(candidate_set))/len(token_set.union(candidate_set))
    return jaccard_distance
            

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
    if not (
        check_dict(vocabulary, str, float, False) and
        (alphabet is None or check_list(alphabet, str, False)) and
        isinstance(first_token, str) and
        method in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]
    ):
        return None
    distance = {}
    if method == "jaccard":
        for key in vocabulary:
            distance_value = calculate_jaccard_distance(first_token, key)
            if distance_value is None:
                return None
            distance[key] = distance_value
    elif method == "frequency-based":
        for key in vocabulary:
            distance[key] = 1.0 - vocabulary[key]
    else:
        return {}
    return distance


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
    if not (
        check_dict(vocabulary, str, float, False) and
        (alphabet is None or check_list(alphabet, str, False)) and
        isinstance(wrong_word, str) and
        method in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]
    ):
        return None
    if calculate_distance is None:
            return None
    if method == "jaccard":
        wrong_word_distance = calculate_distance(wrong_word, vocabulary, "jaccard", None)
    else:
        return None
    if wrong_word_distance == None:
        return None
    min_distance = min(wrong_word_distance.values())
    candidates = [word for word, dis in wrong_word_distance.items() if dis == min_distance]
    if not candidates:
        return None
    if len(candidates) > 1:
        len_differences = {word: len(word) - len(wrong_word) for word in candidates}
        min_len_diff = min(len_differences.values())
        same_len_candidates = [word for word, diff in len_differences.items() if diff == min_len_diff]
        if len(same_len_candidates) > 1:
            closest_word = sorted(same_len_candidates)[0]
        else:
            closest_word = same_len_candidates[0]
    else:
        closest_word = candidates[0]
    return closest_word


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
    if not isinstance(word, str):
        return []
    del_let_candidates = []
    for i in range(len(word)):
        del_let_candidates.append(word[:i] + word[i + 1:])
    return sorted(del_let_candidates)


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
    if not isinstance(word, str):
        return []
    if not check_list(alphabet, str, False):
        return []
    add_let_candidates = []
    for i in range(len(word) + 1):
        for let in alphabet:
            add_let_candidates.append(word[:i] + let + word[i:])
    return sorted(add_let_candidates)


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
    if not isinstance(word, str):
        return []
    if not check_list(alphabet, str, False):
        return []
    replace_let_candidates = []
    for i in range(len(word)):
        for let in alphabet:
            replace_let_candidates.append(word[:i] + let + word[i+1:])
    return sorted(replace_let_candidates)


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
    if not isinstance(word, str):
        return []
    swap_candidates = []
    for i in range(len(word) - 1):
        swap_candidates.append(word[:i] + word[i+1] + word[i] + word[i+2:])
    return sorted(swap_candidates)


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
    if not isinstance(word, str):
        return None
    if not check_list(alphabet, str, True):
        return None
    candidates = (set(delete_letter(word))|
                  set(add_letter(word, alphabet))|
                  set(replace_letter(word, alphabet))|
                  set(swap_adjacent(word)))
    lst_cand = list(candidates)
    for el in lst_cand:
        if lst_cand.count(el) > 1:
            lst_cand.pop(el)
    return sorted(lst_cand)


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
    if not isinstance(word, str):
        return None
    if not check_list(alphabet, str, True):
        return None
    first_cand = (generate_candidates(word, alphabet))
    if first_cand == None:
        return None
    candidates = set(first_cand)
    for token in first_cand:
        second_step_candidates = generate_candidates(token, alphabet)
        if second_step_candidates is None:
            return None
        candidates.update(second_step_candidates)
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
    if not isinstance(word, str):
        return None
    if not check_dict(frequencies, str, float, False):
        return None
    if not check_list(alphabet, str, True):
        return None
    best_cand = {cand: 1.0 for cand in frequencies}
    candidates = propose_candidates(word, alphabet)
    if candidates == None or not candidates:
        return best_cand
    for cand in candidates:
        if cand in frequencies:
            best_cand[cand] = 1 - frequencies[cand]
    return best_cand


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
