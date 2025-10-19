"""
Lab 2.
"""

# pylint:disable=unused-argument
from typing import Literal
from lab_1_keywords_tfidf.main import check_list, check_dict

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
    token_number = len(tokens)
    unique_tokens = set(tokens)
    return {token: tokens.count(token) / token_number for token in unique_tokens}

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
    if not check_list(tokens, str, False) or (
        not check_dict(vocabulary, str, float, False)
    ):
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
    if not isinstance(token, str) or (
        not isinstance(candidate, str)
    ):
        return None
    if not token or not candidate:
        return 1.0
    token_set = set(token)
    candidate_set = set(candidate)
    return 1 - len(token_set & candidate_set) / len(token_set | candidate_set)

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
    if not check_dict(vocabulary, str, float, False) or (
        not isinstance(first_token, str) or
        not isinstance(method, str) or
        not method in [(
            "frequency-based", 
            "jaccard", 
            "levenshtein", 
            "jaro-winkler")]
        ):
        return None
    if method == "frequency-based":
        if alphabet is None:
            return {token: 1.0 for token in vocabulary}
        return calculate_frequency_distance(first_token, vocabulary, alphabet)
    if method == "jaccard":
        dist_method = calculate_jaccard_distance
    elif method == "levenshtein":
        dist_method = calculate_levenshtein_distance
    elif method == "jaro-winkler":
        dist_method = calculate_jaro_winkler_distance
    else:
        return None
    result = {}
    for word in vocabulary:
        distance = dist_method(first_token, word)
        if distance is None:
            return None
        result[word] = float(distance)
    return result

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
    if not isinstance(wrong_word, str) or (
        method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"] or
        not check_dict(vocabulary, str, float, False) or
        not (all(isinstance(key, str)
            and isinstance(value, (int, float)) for key, value in vocabulary.items()))
        ):
        return None
    if alphabet is not None:
        if not check_list(alphabet, str, False) or (
            not all(isinstance(letter, str) for letter in alphabet)
        ):
            return None
    wrong_word_distance = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if wrong_word_distance is None:
        return None
    minimal_distance = min(wrong_word_distance.values())
    candidate_list = [candidate for candidate, value in wrong_word_distance.items()
                      if value == minimal_distance]
    if not candidate_list:
        return None
    if len(candidate_list) > 1:
        min_len_diff = min(abs(len(candidate) - len(wrong_word)) for candidate in candidate_list)
        closest_candidate = [candidate for candidate in candidate_list
                             if abs(len(candidate) - len(wrong_word)) == min_len_diff]
        return sorted(closest_candidate)[0]
    return candidate_list[0]

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
    if not isinstance(token_length, int) or (
        token_length < 0 or
        not isinstance(candidate_length, int) or
        candidate_length < 0
    ):
        return None
    matrix = [[0] * (candidate_length + 1) for _ in range(token_length + 1)]
    for x in range(candidate_length + 1):
        matrix[0][x] = x
    for y in range(token_length + 1):
        matrix[y][0] = y
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
    if not isinstance(token, str) or (
        len(token) < 0 or
        not isinstance(candidate, str) or
        len(candidate) < 0
        ):
        return None
    matrix = initialize_levenshtein_matrix(len(token), len(candidate))
    if matrix is None:
        return None
    cost = 0
    for i in range(1, len(token) + 1):
        for j in range(1, len(candidate) + 1):
            cost = 0 if token[i - 1] == candidate[j - 1] else 1
            deleting = matrix[i - 1][j] + 1
            inserting = matrix[i][j - 1] + 1
            replacing = matrix[i - 1][j - 1] + cost
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
    if not isinstance(token, str) or (
        len(token) < 0 or
        not isinstance(candidate, str) or
        len(candidate) < 0
        ):
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
    result = []
    for i in range(len(word)):
        redacted_word = word[:i] + word[i + 1:]
        result.append(redacted_word)
    return sorted(result)

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
    if not isinstance(word, str) or (
        not check_list(alphabet, str, False)
    ):
        return []
    result = []
    for i in range(len(word) + 1):
        for letter in alphabet:
            result.append(word[:i] + letter + word[i:])
    return sorted(result)

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
    if not isinstance(word, str) or (
        not check_list(alphabet, str, False)
    ):
        return []
    result = []
    for i in range(len(word)):
        for letter in alphabet:
            redacted_word = word[:i] + letter + word[i + 1:]
            result.append(redacted_word)
    return sorted(result)

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
    result = []
    for i in range(len(word) - 1):
        redacted_word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        result.append(redacted_word)
    return sorted(result)

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
    if not isinstance(word, str) or (
        not check_list(alphabet, str, True)
    ):
        return None
    result = []
    result.extend(delete_letter(word))
    if alphabet:
        result.extend(add_letter(word, alphabet))
        result.extend(replace_letter(word, alphabet))
    result.extend(swap_adjacent(word))
    return sorted(set(result))

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
    if not isinstance(word, str) or (
        not check_list(alphabet, str, True)
    ):
        return None
    generated_cands = generate_candidates(word, alphabet)
    if generated_cands is None:
        return None
    candidate_list = set(generated_cands)
    for candidate in generated_cands:
        additional_cands = generate_candidates(candidate, alphabet)
        if additional_cands is None:
            return None
        candidate_list.update(additional_cands)
    return tuple(sorted(candidate_list))

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
    if not isinstance(word, str) or (
        not check_dict(frequencies, str, float, False) or
        not check_list(alphabet, str, True)
    ):
        return None
    proposed_cands = propose_candidates(word, alphabet)
    if not proposed_cands:
        distance_cands = {}
        for token, freq in frequencies.items():
            distance_cands[token] = float(freq) if token == word else 1.0
        return distance_cands
    distance = {}
    for token in frequencies:
        freq_value = frequencies.get(token, 0.0)
        distance[token] = 1.0 - float(freq_value) if token in proposed_cands else 1.0
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
