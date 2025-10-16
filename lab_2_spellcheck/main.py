"""
Lab 2.
"""

# pylint:disable=unused-argument
from typing import Literal

from lab_1_keywords_tfidf.main import (
    check_list,
    check_dict
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

    unique_tokens = set(tokens)
    all_tokens = len(tokens)

    return {token : tokens.count(token) / all_tokens for token in unique_tokens}


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
    In case of both strings being empty, 0.0 is returned.
    """
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None
    if not token or not candidate:
        return 1.0

    tok = set(token)
    cand = set(candidate)
    jaccard_distance = 1 - len(tok & cand)/len(tok | cand)
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
        vocabulary and
        (alphabet is None or check_list(alphabet, str, False)) and
        isinstance(first_token, str) and
        method in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]
    ):
        return None

    if method == "jaccard":
        jaccard_dist = {}
        for key in vocabulary:
            value = calculate_jaccard_distance(key, first_token)
            if value is None:
                return None
            jaccard_dist[key] = value
        return jaccard_dist

    if method == "frequency-based":
        if alphabet is None:
            return {token: 1.0 for token in vocabulary}
        freq_dist = calculate_frequency_distance(first_token, vocabulary, alphabet)
        return freq_dist

    if method == "levenshtein":
        levenshtein_dist = {}
        for word in vocabulary:
            distance = calculate_levenshtein_distance(first_token, word)
            if distance is None:
                return None
            levenshtein_dist[word] = float(distance)
        return levenshtein_dist
    
    if method == "jaro-winkler":
        jaro_winkler_dist = {}
        for word in vocabulary:
            distance = calculate_jaro_winkler_distance(first_token, word)
            if distance is None:
                return None
            jaro_winkler_dist[word] = max(0.0, min(1.0, distance))
        return jaro_winkler_dist
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
    if not (
        isinstance(wrong_word, str) and
        check_dict(vocabulary, str, float, False) and
        method in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"] and
        (alphabet is None or check_list(alphabet, str, False))
    ):
        return None
    distances = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if not distances:
        return None

    min_value = min(distances.values())
    candidates = [candidate for candidate, value in distances.items() if value == min_value]
    if not candidates:
        return None

    if len(candidates) > 1:
        min_length_diff = min(abs(len(candidate) - len(wrong_word)) for candidate in candidates)
        closest_candidates = [candidate for candidate in candidates
                              if abs(len(candidate) - len(wrong_word)) == min_length_diff]
        return sorted(closest_candidates)[0]

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
    if (not isinstance(token_length, int) or
        not isinstance(candidate_length, int) or
        token_length < 0 or
        candidate_length < 0):
        return None
    matrix = []
    for i in range(token_length + 1):
        if i == 0:
            matrix_line = list(range(candidate_length + 1))
            matrix.append(matrix_line)
        else:
            matrix_line = [i if ii == 0 else 0 for ii in range(candidate_length + 1)]
            matrix.append(matrix_line)
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
    matrix = initialize_levenshtein_matrix(len(token), len(candidate))
    if not matrix:
        return None
    for i in range(1, len(token) + 1):
        for j in range(1, len(candidate) + 1):
            if token[i - 1] == candidate[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                delete_cost = matrix[i - 1][j] + 1
                insert_cost = matrix[i][j - 1] + 1
                replace_cost = matrix[i - 1][j - 1] + 1
                matrix[i][j] = min(delete_cost, insert_cost, replace_cost)
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
    if not matrix:
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

    shortened_words = []
    for i in range(len(word)):
        new_word = word[:i] + word[i+1:]
        shortened_words.append(new_word)
    return sorted(shortened_words)


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
    if not isinstance(word, str) or not check_list(alphabet, str, False):
        return []
    extended_words = []
    for i in range(len(word) + 1):
        for letter in alphabet:
            new_word = word[:i] + letter + word[i:]
            extended_words.append(new_word)
    return sorted(extended_words)


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
    if not isinstance(word, str) or not check_list(alphabet, str, False) or word is None:
        return []

    replased_letters_words = []
    for i in range(len(word)):
        for letter in alphabet:
            new_word = word[:i] + letter + word[i + 1:]
            replased_letters_words.append(new_word)
    return sorted(replased_letters_words)


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

    replaced_letters_words = []
    for i in range(len(word) - 1):
        new_word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        replaced_letters_words.append(new_word)
    return sorted(replaced_letters_words)


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
    if not isinstance(word, str) or not check_list(alphabet, str, True):
        return None

    return sorted(
        delete_letter(word) +
        add_letter(word, alphabet) +
        replace_letter(word, alphabet) +
        swap_adjacent(word))


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
    if not isinstance(word, str) or not check_list(alphabet, str, True):
        return None
    candidates = generate_candidates(word, alphabet)
    if candidates is None:
        return None
    all_candidates = set(candidates)
    for token in list(all_candidates):
        second_level_candidates = generate_candidates(token, alphabet)
        if second_level_candidates is None:
            return None
        all_candidates.update(second_level_candidates)
    return tuple(sorted(set(all_candidates)))


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
    if not isinstance(word, str) or not check_dict(frequencies, str, float, False):
        return None
    if not frequencies:
        return {}
    if not check_list(alphabet, str, True):
        return None
    candidates = propose_candidates(word, alphabet)
    candidate_frequencies = {token: 1.0 for token in frequencies.keys()}
    valid_candidates = set()
    if candidates:
        valid_candidates = set(candidates) & set(frequencies.keys())
    for candidate in valid_candidates:
        if candidate in frequencies:
            distance_freq = 1 - frequencies[candidate]
        else: distance_freq = 1.0
        candidate_frequencies[candidate] = distance_freq
    return candidate_frequencies


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
    if (
        not isinstance(token, str) or
        not isinstance(candidate, str) or
        not isinstance(match_distance, int)
        or match_distance < 0
        ):
        return None
    match_count = 0
    if_token_letter_match = [False] * len(token)
    if_candidate_letter_match = [False] * len(candidate)
    for i, el in enumerate(token):
        start = max(0, i - match_distance)
        end = min(len(candidate), i + match_distance + 1)
        for j in range(start, end):
            if not if_candidate_letter_match[j] and el == candidate[j]:
                if_token_letter_match[i] = True
                if_candidate_letter_match[j] = True
                match_count += 1
                break
    
    return (match_count, if_token_letter_match, if_candidate_letter_match)



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
    if (
        not isinstance(token, str) or
        not isinstance(candidate, str) or
        not check_list(token_matches, bool, True) or
        not check_list(candidate_matches, bool, True) or
        len(token_matches) != len(token) or
        len(candidate_matches) != len(candidate)
        ):
        return None

    transpositions = 0
    token_match_els = []
    for i in range(len(token)):
        if token_matches[i]:
            token_match_els.append(token[i])
    if not token_match_els:
        return 0

    candidate_match_els = []
    for j in range(len(candidate)):
        if candidate_matches[j]:
            candidate_match_els.append(candidate[j])
    if not candidate_match_els:
        return 0

    if len(token_match_els) != len(candidate_match_els):
        return 0

    for i in range(len(token_match_els)):
        if token_match_els[i] != candidate_match_els[i]:
            transpositions += 1
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
    if (not isinstance(token, str) or token is None or
        not isinstance(candidate, str) or candidate is None or
        not isinstance(matches, int) or matches < 0 or
        not isinstance(transpositions, int) or transpositions < 0
        ):
        return None

    if matches == 0:
        return 1.0
    else:
        jaro_sim = 1/3 * (matches / len(token) +
                          matches / len(candidate) +
                          (matches - transpositions) / matches
                          )
    jaro_distance = 1 - jaro_sim
    return jaro_distance

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
    if (not isinstance(token, str) or token is None or
        not isinstance(candidate, str) or candidate is None or
        not isinstance(jaro_distance, float) or jaro_distance < 0 or jaro_distance > 1 or
        not isinstance(prefix_scaling, float) or not 1 > prefix_scaling > 0
        ):
        return None
    
    prefix_len = 0
    for i in range(min(4, len(token), len(candidate))):
        if token[i] == candidate[i]:
            prefix_len += 1
        else:
            break
    adjustment = prefix_len * prefix_scaling * jaro_distance
    return adjustment

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
    if (not isinstance(token, str) or
        not isinstance(candidate, str) or
        not isinstance(prefix_scaling, float) or prefix_scaling < 0 or prefix_scaling is None
        ):
        return None
    if len(token) == 0 or len(candidate) == 0:
        return 1.0
    max_len = max(len(token), len(candidate))
    match_distance = (max_len // 2) - 1
    matches = get_matches(token, candidate, match_distance)
    if matches is None:
        return None
    if (not isinstance(matches, tuple) or len(matches) != 3 or
        not isinstance(matches[0], int) or matches[0] < 0 or
        not check_list(matches[1], bool, False) or
        not check_list(matches[2], bool, False)
        ):
        return None
    transpositions = count_transpositions(token, candidate, matches[1], matches[2])
    if transpositions is None:
        return None
    jaro_distance = calculate_jaro_distance(token, candidate, matches[0], transpositions)
    if jaro_distance is None or jaro_distance > 1:
        return None
    adjustment = winkler_adjustment(token, candidate, jaro_distance, prefix_scaling)
    if adjustment is None or adjustment > 1:
        return None
    jaro_winkler_distance = jaro_distance - adjustment
    return max(0, jaro_winkler_distance)
