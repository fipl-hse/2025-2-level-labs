"""
Lab 2.
"""
# pylint:disable=unused-argument
from typing import Literal

from lab_1_keywords_tfidf.main import check_dict, check_list


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
    relative_frequencies = {}
    for token in tokens:
        relative_frequencies[token] = tokens.count(token) / len(tokens)
    return relative_frequencies


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
    out_of_vocab_words = []
    for token in tokens:
        if token not in vocabulary:
            out_of_vocab_words.append(token)
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
    jaccard_distance = 1 - len(set(token) & set(candidate)) / len(set(token) | set(candidate))
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
    distance = {}
    if method == "jaccard":
        for token in vocabulary:
            distance[token] = calculate_jaccard_distance(first_token, token)
            if distance[token] is None:
                return None
    if method == "frequency-based":
        if alphabet is None:
            alphabet = []
        distance = calculate_frequency_distance(first_token, vocabulary, alphabet)
    if method == "levenshtein":
        for token in vocabulary:
            distance[token] = calculate_levenshtein_distance(first_token, token)
            if distance[token] is None:
                return None
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
    if vocabulary == {}:
        return None
    distance_dict = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if distance_dict is None:
        return None
    minimum_distance = min(distance_dict.values())
    maybe_correct_words = [
    key for key, value in distance_dict.items()
    if value == minimum_distance
    ]
    correct_word = maybe_correct_words[0]
    for candidate_word in maybe_correct_words:
        if abs(len(wrong_word) - len(candidate_word)) == abs(len(wrong_word) - len(correct_word)):
            correct_word = min(correct_word, candidate_word)
        else:
            correct_word = min(
                correct_word, candidate_word, key=lambda x: abs(len(wrong_word) - len(x))
                )
    return correct_word


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
    if (not isinstance(token_length, int)
        or not isinstance(candidate_length, int)
        or candidate_length < 0
        or token_length < 0
    ):
        return None
    levenshtein_matrix = [[0 for i in range(candidate_length + 1)] for j in range(token_length + 1)]
    levenshtein_matrix[0] = list(range(candidate_length + 1))
    for i in range(token_length + 1):
        levenshtein_matrix[i][0] = i
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
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None
    levenshtein_matrix = initialize_levenshtein_matrix(len(token), len(candidate))
    if levenshtein_matrix is None:
        return None
    for i in range(1, len(token) + 1): #row index
        for j in range(1, len(candidate) + 1): #column index
            if token[i - 1] == candidate[j - 1]:
                levenshtein_matrix[i][j] = levenshtein_matrix[i - 1][j - 1]
            else:
                levenshtein_matrix[i][j] = min(
                    levenshtein_matrix[i - 1][j] + 1,
                    levenshtein_matrix[i][j - 1] + 1,
                    levenshtein_matrix[i - 1][j - 1] + 1
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
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None
    levenshtein_matrix = fill_levenshtein_matrix(token, candidate)
    if levenshtein_matrix is None:
        return None
    return levenshtein_matrix[len(token)][len(candidate)]


def delete_letter(word: str) -> list[str]:
    """
    Generate all possible words by deleting one letter from the word.

    Args:
        word (str): The input incorrect word.

    Returns:
        list[str]: A sorted list of words with one letter removed at each position.

    In case of corrupt input arguments, empty list is returned.
    """
    del_letter_candidates = []
    if not isinstance(word, str):
        return del_letter_candidates
    for i in range(len(word)):
        del_letter_candidates.append(word[:i] + word[i+1:])
    return sorted(del_letter_candidates)


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
    add_letter_candidates = []
    if not isinstance(word, str) or not check_list(alphabet, str, True):
        return add_letter_candidates
    for i in range(len(word) + 1):
        for letter in alphabet:
            add_letter_candidates.append(word[:i] + letter + word[i:])
    return sorted(add_letter_candidates)


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
    replace_letter_candidates = []
    if not isinstance(word, str) or not check_list(alphabet, str, True):
        return replace_letter_candidates
    for i in range(len(word)):
        for letter in alphabet:
            replace_letter_candidates.append(word[:i] + letter + word[i+1:])
    return sorted(replace_letter_candidates)


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
    swap_adjacent_candidates = []
    if not isinstance(word, str):
        return swap_adjacent_candidates
    for i in range(len(word) - 1):
        swap_adjacent_candidates.append(word[:i] + word[i+1] + word[i] + word[i+2:])
    return sorted(swap_adjacent_candidates)


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
    candidates_list = []
    candidates_list.extend(delete_letter(word))
    candidates_list.extend(add_letter(word, alphabet))
    candidates_list.extend(swap_adjacent(word))
    candidates_list.extend(replace_letter(word, alphabet))
    return sorted(list(set(candidates_list)))


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
    candidates_list = []
    first_operation_candidates = generate_candidates(word, alphabet)
    if first_operation_candidates is None:
        return None
    for operated_word in first_operation_candidates:
        second_operation_candidates = generate_candidates(operated_word, alphabet)
        if second_operation_candidates is None:
            return None
        candidates_list.extend(second_operation_candidates)
    return tuple(sorted(list(set(candidates_list))))


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
    if (not isinstance(word, str)
        or not check_dict(frequencies, str, float, False)
        or not check_list(alphabet, str, True)
    ):
        return None
    frequency_distances = {key: 1.0 for key in frequencies}
    for key in frequency_distances:
        if key is None:
            return None
    candidates_tuple = propose_candidates(word, alphabet)
    if candidates_tuple is None:
        candidates_tuple = ()
    for candidate in candidates_tuple:
        if candidate in frequencies:
            if frequencies[candidate] is None:
                return None
            frequency_distances[candidate] = 1.0 - frequencies[candidate]
            if frequencies[candidate] is None:
                return None
            if frequency_distances[candidate] is None:
                return None
    return frequency_distances


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
