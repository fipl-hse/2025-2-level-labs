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

    return {token: tokens.count(token) / len(tokens) for token in tokens}


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
    if not check_list(tokens, str, False) or not check_dict(vocabulary, str, float, False):
        return None

    return [token for token in tokens if not vocabulary.get(token)]

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

    if not token or not candidate:
        return 1.0

    token_set = set(token)
    candidate_set = set(candidate)
    jaccard_distance = 1 - len(token_set.intersection(candidate_set)) / len(
            token_set.union(candidate_set)
                )

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
    if not (isinstance(first_token, str)
        and check_dict(vocabulary, str, float, False)
        and isinstance(method, str)
        and method in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]
        and (alphabet is None or check_list(alphabet, str, False))):

        return None

    distance = {}

    if method == "jaccard":
        for candidate in vocabulary.keys():
            distance_jaccard = calculate_jaccard_distance(first_token, candidate)
            if distance_jaccard is None:
                return None
            distance[candidate] = distance_jaccard

    if method == "frequency-based":
        if alphabet is None:
            return {key: 1.0 for key in vocabulary.keys()}
        freq_distance = calculate_frequency_distance(first_token, vocabulary, alphabet)
        if freq_distance is None:
            return None
        distance = freq_distance

    if method == "levenshtein":
        for candidate in vocabulary.keys():
            levenshtein_distance = calculate_levenshtein_distance(first_token, candidate)
            if levenshtein_distance is None:
                return None
            distance[candidate] = levenshtein_distance

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
    if not isinstance(wrong_word, str) and check_dict(vocabulary, str, float, False):
        return None
    if not method in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]:
        return None
    if alphabet is None and check_list(alphabet, str, False):
        return None


    calculated_distances = calculate_distance(wrong_word, vocabulary, method, alphabet)

    if calculated_distances is None:
        return None

    best_value = min(calculated_distances.values())

    candidates = [word for word, value in calculated_distances.items() if value == best_value]

    if len(candidates) > 1:
        candidates.sort(key=lambda word: (abs(len(word) - len(wrong_word)), word))

    return candidates[0]

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
    if (not (isinstance(token_length, int) and token_length >= 0) or
        not (isinstance(candidate_length, int) and candidate_length >= 0)):
        return None

    levenshtein_matrix = [[0 for _ in range(candidate_length + 1)] for _ in range(token_length + 1)]

    for i in range(candidate_length + 1):
        levenshtein_matrix[0][i] = i

    for j in range(token_length + 1):
        levenshtein_matrix[j][0] = j

    return levenshtein_matrix

def fill_levenshtein_matrix(token: str, candidate: str) -> list[list[int]] | None:
    """
    Fill a Levenshtein matrix with edit distances between all prefixes.

    Args:
        token (str): First string.
        candidate (str): Second string.

    Returns:
        list[list[int]] | None: Completed Levenshtein distance matrix.
    """
    if not (isinstance(token, str) and isinstance(candidate, str)):
        return None

    levenshtein_matrix = initialize_levenshtein_matrix(len(token), len(candidate))

    if not levenshtein_matrix:
        return None

    for i in range(1, len(token) + 1):
        for j in range(1, len(candidate) + 1):
            cost = 0 if token[i - 1] == candidate[j - 1] else 1

            levenshtein_matrix[i][j] = min(
                levenshtein_matrix[i - 1][j] + 1,
                levenshtein_matrix[i][j - 1] + 1,
                levenshtein_matrix[i - 1][j - 1] + cost,
             )

    return levenshtein_matrix

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

    levenshtein_matrix = fill_levenshtein_matrix(token, candidate)

    if not levenshtein_matrix:
        return None

    return levenshtein_matrix[-1][-1]


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

    return sorted([word[:i] + word[i + 1 :] for i in range(len(word))])

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
    if not (isinstance(word, str) and check_list(alphabet, str, False) and word):
        return []

    return sorted(
    [word[:i] + letter + word[i:]
    for i in range(len(word) + 1) for letter in alphabet]
    )

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
    if not (isinstance(word, str) and check_list(alphabet, str, False) and word):
        return []

    return sorted([word[:i] + letter + word[i + 1:]
        for i in range(len(word))
        for letter in alphabet]
    )

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
    if not (isinstance(word, str) and word):
        return []

    return sorted(
        [word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    for i in range(len(word) - 1)]
    )

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
    if (not isinstance(word, str) or
       not check_list(alphabet, str, True)
        ):
        return None


    if not word:
        return alphabet

    return sorted(
            delete_letter(word)
            + add_letter(word, alphabet)
            + replace_letter(word, alphabet)
            + swap_adjacent(word)
    )

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
    if (
        not isinstance(word, str) and
        not check_list(alphabet, str, True)
        ):
        return None

    candidates = generate_candidates(word, alphabet)

    if not check_list(candidates, str, True):
        return None

    proposed_candidates = set()

    for candidate in candidates:
        generated_candidate = generate_candidates(candidate, alphabet)
        if not check_list(generated_candidate, str, True) or not generated_candidate:
            return None
        proposed_candidates.update(generated_candidate)

    return tuple(sorted(proposed_candidates))

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
    if not (
        isinstance(word, str)
        and check_dict(frequencies, str, float, False)
        and check_list(alphabet, str, True)
    ):
        return None

    frequency_distance = {token: 1.0 for token in frequencies}

    candidates = propose_candidates(word, alphabet)
    if candidates is None:
        return frequency_distance

    for candidate in candidates:
        if frequencies.get(candidate):
            frequency_distance[candidate] = 1.0 - frequencies[candidate]

    return frequency_distance


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
