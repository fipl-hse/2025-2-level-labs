"""
Lab 2.
"""

# pylint:disable=unused-argument
from typing import Literal
from lab_1_keywords_tfidf.main import (check_dict, check_list)


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
    total_tokens = 0
    for token in tokens:
        total_tokens += 1
    count_frequencies = {}
    for token in tokens:
        if token in count_frequencies:
            count_frequencies[token] += 1
        else:
            count_frequencies[token] = 1
    relative_frequencies = {}
    for token, count in count_frequencies.items():
        relative_frequencies[token] = count / total_tokens
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
    if not check_list(tokens, str, False):
        return None
    if not check_dict(vocabulary, str, float, False):
        return None
    out_of_vocab= []
    for token in tokens:
        if token not in vocabulary.keys():
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
    if token == "" and candidate == "":
        return 1.0
    token_in_set = set(token)
    candidate_in_set = set(candidate)
    intersection = token_in_set.intersection(candidate_in_set)
    union = token_in_set.union(candidate_in_set)
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
    if not isinstance(first_token, str):
        return None
    if not check_dict(vocabulary, str, float, False):
        return None
    if alphabet is not None and not check_list(alphabet, str, True):
        return None
    if method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]:
        return None
    distance_score = {}
    if method == "jaccard":
        for word in vocabulary:
            jaccard_distance = calculate_jaccard_distance(first_token, word)
            distance_score[word] = jaccard_distance
    elif method == "frequency-based":
        freq_distance = calculate_frequency_distance(first_token, vocabulary, alphabet)
        if freq_distance is None:
            return None
        for word in vocabulary:
            distance_score[word] = freq_distance.get(word, 1.0)
    else:
        for word in vocabulary:
            distance_score[word] = 0.0
    return distance_score


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
    if not vocabulary:
        return None
    distance_score = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if not distance_score:
        return None
    min_distance = min(distance_score.values())
    candidates = []
    for word, distance in distance_score.items():
        if distance == min_distance:
            candidates.append(word)
    if not candidates:
        return None
    length_wrond_word = len(wrong_word)
    differences = []
    for word in candidates:
        diff = abs(len(word)) - length_wrond_word
        differences.append(diff)
    min_difference = min(differences)
    good_candidates = []
    for word in candidates:
        if abs(len(word)) - length_wrond_word == min_difference:
            good_candidates.append(word)
    if not good_candidates:
        return None
    best_candidate = min(good_candidates)
    return best_candidate


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
    candidates = []
    for i in range(len(word)):
        new = word[:i] + word[i+1:]
        candidates.append(new)
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
    if not isinstance(word, str):
        return []
    if not check_list(alphabet, str, True):
        return []
    candidates = []
    for i in range(len(word)+1):
        for letter in alphabet:
            new = word[:i] + letter + word[i:]
            candidates.append(new)
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
    if not isinstance(word, str):
        return []
    if not check_list(alphabet, str, True):
        return []
    candidates = []
    for i in range(len(word)):
        for letter in alphabet:
            new = word[:i] + letter + word[i+1:]
            candidates.append(new)
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
    for i in range(len(word)-1):
        new = word[:i] + word[i+1] + word[i] + word[i+2:]
        candidates.append(new)
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
    if not isinstance(word, str):
        return None
    if not check_list(alphabet, str, True):
        return None
    candidates = []
    del_candidates = delete_letter(word)
    candidates.extend(del_candidates)
    add_candidates = add_letter(word, alphabet)
    candidates.extend(add_candidates)
    replace_candidates = replace_letter(word, alphabet)
    candidates.extend(replace_candidates)
    swap_candidates = swap_adjacent(word)
    candidates.extend(swap_candidates)
    candidates_set = set(candidates)
    candidates_set.discard("")
    return sorted(list(candidates_set))


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
    candidates = generate_candidates(word, alphabet)
    if candidates is None:
        return None
    return tuple(candidates)


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
    if not check_list(alphabet, str, True):
        return None
    if not isinstance(frequencies, dict):
        return None
    candidates = propose_candidates(word, alphabet)
    if candidates is None:
        return None
    distance_freq = {}
    max_freq = max(frequencies.values())
    for candidate in candidates:
        freq = frequencies.get(candidate, 0.0)
        distance_freq[candidate] = 1 - (freq / max_freq)
    return distance_freq


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


def winkler_adjustment_distance(
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
        float | None: Jaro-Winkler distance score.
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
    """
