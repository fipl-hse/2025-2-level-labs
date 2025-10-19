"""
Lab 2.
"""

# pylint:disable=unused-argument
from typing import Literal
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
    if not user_input:
        return can_be_empty
    return all(isinstance(element, elements_type) for element in user_input)


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
    if not user_input:
        return can_be_empty
    return (all(isinstance(key, key_type) for key in user_input) and
        all(isinstance(value, value_type) for value in user_input.values()))

def check_non_negative_int(user_input: Any) -> bool:
    """
    Check if the object is a non-negative integer (not bool).

    Args:
        user_input (Any): Object to check

    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(user_input, int) and not isinstance(user_input, bool) and user_input >= 0


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
    for element in tokens:
        vocabulary[element] = tokens.count(element)/len(tokens)
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
    if (not check_list(tokens, str, False) or
    not check_dict(vocabulary, str, float, False)):
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
    if not token and not candidate:
        return 1.0
    set1 = set(token)
    set2 = set(candidate)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 1.0
    return 1 - len(intersection) / len(union)


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
    if (not isinstance(first_token, str) or not first_token
    or not check_dict(vocabulary, str, float, False) or
    method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]):
        return None
    distances = {}
    if method == "jaccard":
        for word in vocabulary:
            distance = calculate_jaccard_distance(first_token, word)
            if distance is None:
                return None
            distances[word] = distance
    if method == "frequency-based":
        freq_distances = calculate_frequency_distance(first_token, vocabulary, alphabet or [])
        if freq_distances is None:
            return None
        return freq_distances
    if method == "levenshtein":
        for word in vocabulary:
            distance = calculate_levenshtein_distance(first_token, word)
            if distance is None:
                return None
            distances[word] = float(distance)
    if method == "jaro-winkler":
        for word in vocabulary:
            distance = calculate_jaro_winkler_distance(first_token, word)
            if distance is None:
                return None
            distances[word] = distance
    return distances if distances else None


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
    if not isinstance(wrong_word, str) or not wrong_word:
        return None
    if not check_dict(vocabulary, str, float, False):
        return None
    if method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]:
        return None
    if alphabet is not None and not check_list(alphabet, str, True):
        return None
    distances = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if not distances:
        return None
    min_distance = min(distances.values())
    candidates = [word for word, dist in distances.items() if dist == min_distance]
    candidates.sort(key=lambda x: (abs(len(x) - len(wrong_word)), x))
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
    if not check_non_negative_int(token_length) or not check_non_negative_int(candidate_length):
        return None
    matrix = [[0] * (candidate_length + 1) for _ in range(token_length + 1)]
    for i in range(token_length + 1):
        matrix[i][0] = i
    for j in range(candidate_length + 1):
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
    matrix = initialize_levenshtein_matrix(len(token), len(candidate))
    if matrix is None:
        return None
    for i in range(1, len(token) + 1):
        for j in range(1, len(candidate) + 1):
            cost = 0 if token[i-1] == candidate[j-1] else 1
            matrix[i][j] = min(
                matrix[i-1][j] + 1,
                matrix[i][j-1] + 1,
                matrix[i-1][j-1] + cost
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
    candidates = []
    for i in range(len(word)):
        candidate = word[:i] + word[i+1:]
        candidates.append(candidate)
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
    if not isinstance(word, str) or not isinstance(alphabet, list):
        return []
    if not all(isinstance(letter, str) for letter in alphabet):
        return []
    candidates = []
    for i in range(len(word) + 1):
        for letter in alphabet:
            candidate = word[:i] + letter + word[i:]
            candidates.append(candidate)
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
    if not isinstance(word, str) or not isinstance(alphabet, list):
        return []
    if not all(isinstance(letter, str) for letter in alphabet):
        return []
    if not word:
        return []
    candidates = []
    for i, current_letter in enumerate(word):
        for letter in alphabet:
            if letter != current_letter:
                candidate = word[:i] + letter + word[i+1:]
                candidates.append(candidate)
    return sorted(candidates)


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
    candidates = []
    for i in range(len(word) - 1):
        candidate = word[:i] + word[i+1] + word[i] + word[i+2:]
        candidates.append(candidate)
    return sorted(candidates)


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
    all_candidates = set()
    delete_candidates = delete_letter(word)
    add_candidates = add_letter(word, alphabet)
    replace_candidates = replace_letter(word, alphabet)
    swap_candidates = swap_adjacent(word)
    all_candidates.update(delete_candidates, add_candidates, replace_candidates, swap_candidates)
    return sorted(list(all_candidates))


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
    candidates = set()
    first_step_candidates = generate_candidates(word, alphabet)
    if first_step_candidates is None:
        return None
    candidates.update(first_step_candidates)
    for candidate_word in first_step_candidates:
        second_step_candidates = generate_candidates(candidate_word, alphabet)
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
    if not isinstance(word, str) or not check_dict(frequencies, str, (int, float), False):
        return None
    if not check_list(alphabet, str, True):
        return None
    proposed = propose_candidates(word, alphabet)
    candidates = set(proposed) if proposed is not None else set()
    distances = {}
    for vocab_word, frequency in frequencies.items():
        distance = 1.0 - float(frequency) if vocab_word in candidates else 1.0
        distances[vocab_word] = distance
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
    token_len = len(token)
    candidate_len = len(candidate)
    token_matches = [False] * token_len
    candidate_matches = [False] * candidate_len
    matches = 0
    for i in range(token_len):
        start = max(0, i - match_distance)
        end = min(candidate_len, i + match_distance + 1)
        for j in range(start, end):
            if not candidate_matches[j] and token[i] == candidate[j]:
                token_matches[i] = True
                candidate_matches[j] = True
                matches += 1
                break
    return matches, token_matches, candidate_matches


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
    token_matched_chars = []
    candidate_matched_chars = []
    for i, matched in enumerate(token_matches):
        if matched:
            token_matched_chars.append(token[i])
    for j, matched in enumerate(candidate_matches):
        if matched:
            candidate_matched_chars.append(candidate[j])
    transpositions = 0
    min_len = min(len(token_matched_chars), len(candidate_matched_chars))
    for k in range(min_len):
        if token_matched_chars[k] != candidate_matched_chars[k]:
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
    if (not isinstance(token, str) or not isinstance(candidate, str) or
        not isinstance(matches, int) or not isinstance(transpositions, int)):
        return None
    token_len = len(token)
    candidate_len = len(candidate)
    if matches < 0 or transpositions < 0:
        return None
    if transpositions > matches:
        return None
    if matches > 0 and (token_len == 0 or candidate_len == 0):
        return None
    if matches == 0:
        return 1.0
    jaro_similarity = (
        matches / token_len +
        matches / candidate_len +
        (matches - transpositions) / matches
    ) / 3.0
    return 1.0 - jaro_similarity


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
    prefix_length = 0
    for i in range(min(4, len(token), len(candidate))):
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
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None
    if not token and not candidate:
        return 1.0
    match_result = get_matches(token, candidate, max(len(token), len(candidate)) // 2 - 1)
    if match_result is None:
        return None
    matches, token_matches, candidate_matches = match_result
    if matches == 0:
        return 1.0
    transpositions = count_transpositions(token, candidate, token_matches, candidate_matches)
    if transpositions is None:
        return None
    jaro_distance = calculate_jaro_distance(token, candidate, matches, transpositions)
    if jaro_distance is None:
        return None
    winkler_adjustment_value = winkler_adjustment(token, candidate, jaro_distance, prefix_scaling)
    if winkler_adjustment_value is None:
        return None
    jaro_winkler_distance = jaro_distance - winkler_adjustment_value
    return max(0.0, jaro_winkler_distance)
