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
    all_tokens = len(tokens)
    vocab = {}
    for token in tokens:
        vocab[token] = tokens.count(token) / all_tokens
    return vocab


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
    out_of_vocab = []
    for token in tokens:
        if token not in vocabulary:
            out_of_vocab.append(token)
    return out_of_vocab


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
    if len(token) == 0 and len(candidate) == 0:
        return 1.0
    set_token = set(token)
    set_candidate = set(candidate)
    jaccard_distance = (1 - len(set_token & set_candidate) /
                        len(set_token | set_candidate))
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
    if (
        not isinstance(first_token, str)
        or not check_dict(vocabulary, str, float, False)
        or method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]
        or not alphabet is None and not check_list(alphabet, str, False)
    ):
        return None
    calculated_distance = {}
    if method == "jaccard":
        for token in vocabulary:
            distance = calculate_jaccard_distance(first_token, token)
            if distance is None:
                return None
            calculated_distance[token] = distance
    elif method == "levenshtein":
        for token in vocabulary:
            distance = calculate_levenshtein_distance(first_token, token)
            if distance is None:
                return None
            calculated_distance[token] = distance
    elif method == "frequency-based":
        result = calculate_frequency_distance(first_token, vocabulary, alphabet or [])
        if result is None:
            return None
        calculated_distance = result
    elif method == "jaro-winkler":
        return None
    return calculated_distance


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
    if (not isinstance(wrong_word, str) or
        not check_dict(vocabulary, str, float, False) or
        alphabet is not None and not check_list(alphabet, str, True) or
        method not in ("jaccard",
                       "frequency-based",
                       "levenshtein",
                       "jaro-winkler")
    ):
        return None
    if not vocabulary:
        return None
    distances = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if distances is None:
        return None
    lowest_score = min(distances.values())
    candidates = []
    for key, value in distances.items():
        if value == lowest_score:
            candidates.append(key)
    if not candidates:
        return None
    min_length_diff = min(abs(len(candidate) - len(wrong_word)) for candidate in candidates)
    length_candidates = [candidate for candidate in candidates
                        if abs(len(candidate) - len(wrong_word)) == min_length_diff]
    return sorted(length_candidates)[0]


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
    if (not isinstance(token_length, int) or
        not isinstance(candidate_length, int) or
        token_length < 0 or
        candidate_length < 0):
        return None
    matrix = []
    for i in range(token_length + 1):
        row = []
        for j in range(candidate_length + 1):
            if i == 0:
                row.append(j)
            elif j == 0:
                row.append(i)
            else:
                row.append(0)
        matrix.append(row)
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
    if(not isinstance(token, str) or
       not isinstance(candidate, str)):
        return None
    token_length = len(token)
    candidate_length = len(candidate)
    matrix = initialize_levenshtein_matrix(token_length, candidate_length)
    if matrix is None:
        return None
    for i in range(1, token_length + 1):
        for j in range(1, candidate_length + 1):
            cost = 0 if token[i-1] == candidate[j-1] else 1
            delete = matrix[i - 1][j] + 1
            insert = matrix[i][j - 1] + 1
            replace = matrix[i-1][j-1] + cost
            matrix[i][j] = min(delete, insert, replace)
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
    if(not isinstance(token, str) or
       not isinstance(candidate, str)):
        return None
    matrix = fill_levenshtein_matrix(token, candidate)
    if matrix is None:
        return None
    return matrix[len(token)][len(candidate)]


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
    deleted_letter_lst = []
    for i in range(len(word)):
        new = word[:i] + word[i+1:]
        deleted_letter_lst.append(new)
    return sorted(deleted_letter_lst)


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
    if (not isinstance(word, str) or
        not check_list(alphabet, str, False)
    ):
        return []
    added_letter_lst = []
    for i in range(len(word) + 1):
        for letter in alphabet:
            new = word[:i] + letter + word[i:]
            added_letter_lst.append(new)
    return sorted(added_letter_lst)


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
    if (not isinstance(word, str) or
        not check_list(alphabet, str, False)
    ):
        return []
    replaced_letter_lst = []
    for i in range(len(word)):
        for letter in alphabet:
            new = word[:i] + letter + word[i+1:]
            replaced_letter_lst.append(new)
    return sorted(replaced_letter_lst)


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
    swapped = []
    for i in range(len(word) - 1):
        new = word[:i] + word[i+1] + word[i] + word[i+2:]
        swapped.append(new)
    return sorted(swapped)


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
    candidates = (delete_letter(word) + add_letter(word, alphabet)
                         + replace_letter(word, alphabet) + swap_adjacent(word))
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
    if (not isinstance(word, str) or
        not check_list(alphabet, str, True)
    ):
        return None
    candidates = set()
    one_mist_candidates = generate_candidates(word, alphabet)
    if one_mist_candidates is None:
        return None
    candidates.update(one_mist_candidates)
    for candidate in one_mist_candidates:
        two_mist_candidates = generate_candidates(candidate, alphabet)
        if two_mist_candidates is None:
            return None
        candidates.update(two_mist_candidates)
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
    if (not isinstance(word, str) or
        not check_dict(frequencies, str, float, False) or
        not check_list(alphabet, str, True)):
        return None
    distance = {candidate: 1.0 for candidate in frequencies}
    candidates = propose_candidates(word, alphabet)
    if candidates:
        for candidate in candidates:
            if candidate in frequencies:
                distance[candidate] = 1.0 - frequencies.get(candidate, 0.0)
    return distance


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
