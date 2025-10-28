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
    if not check_list(tokens, str,  False):
        return None
    vocabulary = {}
    total_tokens = len(tokens)
    for token in tokens:
        vocabulary[token] = vocabulary.get(token, 0) + 1
    for token in vocabulary:
        vocabulary[token] /= total_tokens
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
    return [token for token in tokens if token not in vocabulary]


def calculate_jaccard_distance(token: str, candidate: str) -> float | None:
    """
    Calculate Jaccard distance between two strings.

    Args:
        token (str): First string to compare.
        candidate (str): Second string to compare.

    Returns:
        float | None: Jaccard distance score in range [0, 1].

    In case of corrupt input arguments, None is returned.
    In case of both strings being empty, 1.0 is returned.
    """
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None
    if token == "" and candidate == "":
        return 1.0
    set_token = set(token)
    set_candidate = set(candidate)
    union = set_token.union(set_candidate)
    if not union:
        return 1.0

    intersection = set_token.intersection(set_candidate)
    jaccard_distance = 1 - len(intersection) / len(union)
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
        not isinstance(first_token, str) or not first_token 
        or not check_dict(vocabulary, str, float, False) 
        or not isinstance(method, str) 
        or method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]
    ):
        return None
    if method == "frequency-based":
        if not check_list(alphabet, str, True) or alphabet is None:
            return {word: 1.0 for word in vocabulary}
        return calculate_frequency_distance(first_token, vocabulary, alphabet)
    distances = {}
    for word in vocabulary:
        dist = None
        if method == "jaccard":
            dist = calculate_jaccard_distance(first_token, word)
        elif method == "levenshtein":
            dist = calculate_levenshtein_distance(first_token, word)
        elif method == "jaro-winkler":
            dist = calculate_jaro_winkler_distance(first_token, word)
        if dist is not None:
            distances[word] = dist
    if not distances:
        return None
    return distances


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
    if not isinstance(wrong_word, str):
        return None
    if not check_dict(vocabulary, str, float, False):
        return None
    if not isinstance(method, str):
        return None
    if method not in ("jaccard", "frequency-based", "levenshtein", "jaro-winkler"):
        return None
    if not check_list(alphabet, str, True):
        return None
    if wrong_word in vocabulary and method != "frequency-based":
        return wrong_word
    distances = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if distances is None:
        return None
    min_distance = min(distances.values())
    candidates = [word for word, dist in distances.items() if dist == min_distance]
    candidates.sort(key=lambda w: (abs(len(w) - len(wrong_word)), w))
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
    if not isinstance(token_length, int) or token_length < 0:
        return None
    if not isinstance(candidate_length, int) or candidate_length < 0:
        return None
    matrix = [[0] * (candidate_length + 1) for _ in range(token_length + 1)]
    for i in range(token_length + 1):
        matrix[i][0] = i
    for j in range(1, candidate_length + 1):
        matrix[0][j] = j
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
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None
    m, n = len(token), len(candidate)
    matrix = initialize_levenshtein_matrix(m, n)
    if matrix is None:
        return None
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if token[i - 1] == candidate[j - 1] else 1
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
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None
    matrix = fill_levenshtein_matrix(token, candidate)
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
    if not isinstance(word, str):
        return []
    changed_words = []
    for letter in range(len(word)):
        new_word = word[:letter] + word[letter+1:]
        changed_words.append(new_word)
    return sorted(changed_words)


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
    changed_words = []
    for i in range(len(word) + 1):
        for letter in alphabet:
            new_word = word[:i] + letter + word[i:]
            changed_words.append(new_word)
    return sorted(changed_words)


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
    changed_words = set()
    for i, original_letter in enumerate(word):
        for letter in alphabet:
            if letter != original_letter:
                new_word = word[:i] + letter + word[i+1:]
                changed_words.add(new_word)
    return sorted(changed_words)


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
    changed_words = []
    length = len(word)
    for i in range(length - 1):
        new_word = word[:i] + word[i+1] + word[i] + word[i+2:]
        changed_words.append(new_word)
    return sorted(changed_words)


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
    candidates = set()
    candidates.update(swap_adjacent(word))
    candidates.update(replace_letter(word, alphabet))
    candidates.update(add_letter(word, alphabet))
    candidates.update(delete_letter(word))
    return sorted(candidates)


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
    all_candidates = set()
    first_level = generate_candidates(word, alphabet)
    if first_level is None:
        return None
    all_candidates.update(first_level)
    for candidate in first_level:
        second_level = generate_candidates(candidate, alphabet)
        if second_level is None:
            return None
        all_candidates.update(second_level)
    return tuple(sorted(all_candidates))


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
    if not isinstance(frequencies, dict) or not frequencies:
        return None
    if not check_list(alphabet, str, True):
        return None
    for token, freq in frequencies.items():
        if not isinstance(token, str) or not isinstance(freq, (int, float)):
            return None
    distances = {token: 1.0 for token in frequencies}
    candidates = propose_candidates(word, alphabet)
    if candidates is None or not candidates:
        return distances
    for candidate in candidates:
        if candidate in frequencies:
            distances[candidate] = 1 - frequencies[candidate]
    return distances


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
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None
    if not isinstance(match_distance, int) or match_distance < 0:
        return None
    len_token = len(token)
    len_candidate = len(candidate)
    token_matches = [False] * len_token
    candidate_matches = [False] * len_candidate
    matches_count = 0
    for i in range(len_token):
        start = max(0, i - match_distance)
        end = min(len_candidate, i + match_distance + 1)
        for j in range(start, end):
            if not candidate_matches[j] and token[i] == candidate[j]:
                token_matches[i] = True
                candidate_matches[j] = True
                matches_count += 1
                break
    return matches_count, token_matches, candidate_matches


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
    if not check_list(token_matches, bool, False):
        return None
    if not check_list(candidate_matches, bool, False):
        return None
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None
    transpositions = 0
    candidate_index = 0
    for i, char in enumerate(token):
        if token_matches[i]:
            while candidate_index < len(candidate) and not candidate_matches[candidate_index]:
                candidate_index += 1
            if candidate_index < len(candidate):
                if char != candidate[candidate_index]:
                    transpositions += 1
                candidate_index += 1
    return transpositions // 2


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
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None
    if not isinstance(matches, int) or not isinstance(transpositions, int):
        return None
    if matches < 0 or transpositions < 0:
        return None
    if matches == 0:
        return 1.0
    m_over_len1 = matches / len(token)
    m_over_len2 = matches / len(candidate)
    mt_over_m = (matches - transpositions) / matches
    jaro_similarity = 1 - (m_over_len1 + m_over_len2 + mt_over_m) / 3
    return jaro_similarity


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
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None
    if not isinstance(jaro_distance, float) or not isinstance(prefix_scaling, float):
        return None
    prefix_len = 0
    min_len = min(len(token), len(candidate), 4)
    while prefix_len < min_len and token[prefix_len] == candidate[prefix_len]:
        prefix_len += 1
    prefix_len = min(4, prefix_len)
    adjusted = prefix_len * prefix_scaling * jaro_distance
    return adjusted


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
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None
    if not isinstance(prefix_scaling, float):
        return None
    if prefix_scaling < 0 or prefix_scaling > 1:
        return None
    match_distance = max(len(token), len(candidate)) // 2 - 1
    match_distance = max(match_distance, 0)
    matches_result = get_matches(token, candidate, match_distance)
    if matches_result is None:
        return None
    matches, token_matches, candidate_matches = matches_result
    if matches == 0:
        return 1.0
    transpositions_result = count_transpositions(token, candidate, token_matches, candidate_matches)
    if transpositions_result is None:
        return None
    jaro = calculate_jaro_distance(token, candidate, matches, transpositions_result)
    if jaro is None:
        return None
    winkler = winkler_adjustment(token, candidate, jaro, prefix_scaling)
    if winkler is None:
        return None
    result = jaro - winkler
    return result
