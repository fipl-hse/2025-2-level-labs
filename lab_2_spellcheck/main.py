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
    amount_tokens = len(tokens)
    return {token: tokens.count(token) /
            amount_tokens for token in tokens}


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
    if (not check_list(tokens, str, False) or
        not check_dict(vocabulary, str, float, False)):
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
    if (not isinstance(token, str) or
        not isinstance(candidate, str)):
        return None
    if not token or not candidate:
        return 1.0
    token_set = set(token)
    candidate_set = set(candidate)
    return (1 - (len(token_set & candidate_set)) /
            len(token_set | candidate_set))


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
    if (not isinstance(first_token, str) or
        not first_token or
        not check_dict(vocabulary, str, float, False) or
        not isinstance(method, str) or
        method not in ("jaccard",
                       "frequency-based",
                       "levenshtein",
                       "jaro-winkler")):
        return None
    if method == 'frequency-based':
        if not check_list(alphabet, str, True) or alphabet is None:
            return {word: 1.0 for word in vocabulary}
        distance = calculate_frequency_distance(first_token, vocabulary, alphabet)
        if distance is None:
            return None
        return distance
    calculated_distance_score = {}
    for word in vocabulary:
        if method == 'jaccard':
            distance_value = calculate_jaccard_distance(first_token, word)
        elif method == 'levenshtein':
            distance_value = calculate_levenshtein_distance(first_token, word)
        elif method == 'jaro-winkler':
            distance_value = calculate_jaro_winkler_distance(first_token, word)
        if distance_value is None:
            return None
        calculated_distance_score[word] = distance_value
    return calculated_distance_score


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
    if (not isinstance(wrong_word, str) or
        not check_dict(vocabulary, str, float, False) or
        not isinstance(method, str) or
        method not in ("jaccard",
                       "frequency-based",
                       "levenshtein",
                       "jaro-winkler") or
        not check_list(alphabet, str, True)):
        return None
    distance_wrong_word_dict = calculate_distance(wrong_word, vocabulary,
                                                      method, alphabet)
    if distance_wrong_word_dict is None:
        return None
    return sorted(distance_wrong_word_dict.items(), key=lambda item:
                  (item[1], abs(len(wrong_word) - len(item[0])), item[0]))[0][0]


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
        token_length < 0 or
        not isinstance(candidate_length, int) or
        candidate_length < 0):
        return None
    matrix = [[0] * (candidate_length + 1) for _ in range(token_length + 1)]
    for i in range(candidate_length + 1):
        matrix[0][i] = i
    for j in range(token_length + 1):
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
    if (not isinstance(token, str) or
        len(token) < 0 or
        not isinstance(candidate, str) or
        len(candidate) < 0):
        return None
    matrix = initialize_levenshtein_matrix(len(token), len(candidate))
    if matrix is None:
        return None
    cost = 0
    for i in range(1, len(token) + 1):
        for j in range(1, len(candidate) + 1):
            cost = 0 if token[i-1] == candidate[j-1] else 1
            deleting = matrix[i - 1][j] + 1
            inserting = matrix[i][j - 1] + 1
            replacing = matrix[i-1][j-1] + cost
            matrix[i][j] = min(deleting, inserting, replacing)
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
    if (not isinstance(token, str) or
        len(token) < 0 or
        not isinstance(candidate, str) or
        len(candidate) < 0):
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
    return sorted([word[:i] + word[i+1:] for i in range(len(word))])


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
        not check_list(alphabet, str, False)):
        return []
    return sorted([word[:i] + letter + word[i:]
                   for letter in alphabet
                   for i in range(len(word)+1)])


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
        not check_list(alphabet, str, False)):
        return []
    return sorted([word[:i] + letter + word[i+1:]
                   for letter in alphabet
                   for i in range(len(word))
                   if letter != word[i]])


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
    return sorted([word[:i] + word[i+1] + word[i] + word[i+2:]
                   for i in range(len(word)-1)])


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
        not check_list(alphabet, str, True)):
        return None
    candidates = []
    candidates.extend(delete_letter(word))
    candidates.extend(add_letter(word, alphabet))
    candidates.extend(replace_letter(word, alphabet))
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
    if (not isinstance(word, str) or
        not check_list(alphabet, str, True)):
        return None
    candidates = set()
    first_step_candidates = generate_candidates(word, alphabet)
    if first_step_candidates is None:
        return None
    candidates.update(first_step_candidates)
    for token in first_step_candidates:
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
    if (not isinstance(word, str) or
        not isinstance(frequencies, dict) or
        not frequencies or
        not check_list(alphabet, str, True)):
        return None
    for token, freq in frequencies.items():
        if not isinstance(token, str) or not isinstance(freq, (int, float)):
            return None
    result = {token: 1.0 for token in frequencies}
    candidates = propose_candidates(word, alphabet)
    if candidates is None or not candidates:
        return result
    for candidate in candidates:
        if candidate in frequencies:
            result[candidate] = 1 - frequencies[candidate]
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
    if (
        not isinstance(token, str) or
        not isinstance(candidate, str) or
        not isinstance(match_distance, int) or
        match_distance < 0
    ):
        return None
    matching_letters = 0
    token_matches = [False] * len(token)
    candidate_matches = [False] * len(candidate)
    for index, token_char in enumerate(token):
        start = max(0, index - match_distance)
        end = min(len(candidate) - 1, index + match_distance)
        for k in range(start, end + 1):
            if not candidate_matches[k] and candidate[k] == token_char:
                matching_letters += 1
                token_matches[index] = True
                candidate_matches[k] = True
                break
    return (matching_letters, token_matches, candidate_matches)


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
    if (not isinstance(token, str) or
        not isinstance(candidate, str) or
        not check_list(token_matches, bool, False) or
        not check_list(candidate_matches, bool, False)):
        return None
    mismatches = 0
    token_chars = [token[index] for index, char
                   in enumerate(token_matches) if char]
    candidate_chars = [candidate[index] for index, char
                       in enumerate(candidate_matches) if char]
    for index, word in enumerate(token_chars):
        if word != candidate_chars[index]:
            mismatches += 1
    return mismatches // 2


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
    if (not isinstance(token, str) or
        not isinstance(candidate, str) or
        not isinstance(matches, int) or
        not isinstance(transpositions, int)):
        return None
    if (matches < 0 or
        transpositions < 0):
        return None
    if matches == 0:
        return 1.0
    return (1 - ((matches / len(token) +
                       matches / len(candidate) +
                       (matches - transpositions) / matches) / 3))


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
    if (not isinstance(token, str) or
        not isinstance(candidate, str) or
        not isinstance(jaro_distance, float) or
        not isinstance(prefix_scaling, float)):
        return None
    max_prefix_length = min(len(token), len(candidate), 4)
    prefix_length = 0
    for i in range(max_prefix_length):
        if token[i] == candidate[i]:
            prefix_length += 1
        else:
            break
    return prefix_length * prefix_scaling * jaro_distance


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
        not isinstance(candidate, str)):
        return None
    if (not isinstance(prefix_scaling, float) or
        prefix_scaling < 0 or
        prefix_scaling > 1):
        return None
    match_distance = max(len(token), len(candidate)) // 2 - 1
    match_distance = max(match_distance, 0)
    matches = get_matches(token, candidate, match_distance)
    if matches is None:
        return None
    amount_matches, token_matches, candidate_matches = matches
    if amount_matches == 0:
        return 1.0
    transpositions = count_transpositions(token, candidate, token_matches, candidate_matches)
    if transpositions is None:
        return None
    jaro_distance = calculate_jaro_distance(token, candidate, amount_matches, transpositions)
    if jaro_distance is None:
        return None
    adjustment = winkler_adjustment(token, candidate, jaro_distance, prefix_scaling)
    if adjustment is None:
        return None
    return jaro_distance - adjustment
