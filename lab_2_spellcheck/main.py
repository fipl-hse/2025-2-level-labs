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
    all_tokens_count = len(tokens)
    token_counts = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    return {token: token_count/all_tokens_count for token, token_count in token_counts.items()}

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
    words_not_in_dictionary = []
    for token in tokens:
        if not token in vocabulary:
            words_not_in_dictionary.append(token)
    return words_not_in_dictionary

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
    if token == '' and candidate == '':
        return 1.0
    tokens_intersection = len(set(token).intersection(set(candidate)))
    tokens_union = len(set(token).union(set(candidate)))
    return 1-(tokens_intersection/tokens_union)

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
    if method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]:
        return None
    if not check_dict(vocabulary, str, float, False):
        return None
    if not check_list(alphabet, str, False) and alphabet != None:
        return None
    if method == 'jaccard':
        distance = {}
        for key,value in vocabulary.items():
            freq_distance = calculate_jaccard_distance(first_token,key)
            if freq_distance is None:
                return None
            distance[key] = distance.get(key, 0) + freq_distance
        return distance
    if method == 'frequency-based':
        if alphabet is None:
            return {token: 1.0 for token in vocabulary}
        return calculate_frequency_distance(first_token, vocabulary, alphabet)

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

    if not isinstance(wrong_word, str):
        return None
    if method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]:
        return None
    if not check_dict(vocabulary, str, float, False):
        return None
    if not check_list(alphabet, str, False) and alphabet != None:
        return None
    # distance = calculate_distance(wrong_word, vocabulary, method, alphabet)
    # if distance is None:
    #     return None
    # min_distance = 1.0 
    # for key,value in distance.items():
    #     min_distance = min(min_distance,value)
    # candidates = []
    # for key,value in distance.items():
    #     if value == min_distance:
    #         candidates.append(key)
    # min_diff = 100000
    # for candidate in candidates:
    #     min_diff = min(min_diff, abs(len(wrong_word) - len(candidate)))
    # for candidate in candidates:
    #     if abs(len(wrong_word) - len(candidate)) != min_diff:
    #         candidates.remove(candidate)
    # candidates = sorted(candidates)
    # return candidates[0]
    distances = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if not distances:
        return None
    min_distance = min(distances.values())
    candidates = [token for token, token_distance in distances.items()
                  if token_distance == min_distance]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    min_length_differences = min(len(candidate) - len(wrong_word) for candidate in candidates)
    min_length_candidates = [candidate for candidate in candidates
                                if len(candidate) - len(wrong_word) == min_length_differences]
    return sorted(min_length_candidates)[0]

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
    n = len(word)
    candidates_without_letter = []
    for i in range(n):
        candidates_without_letter.append(word[:i] + word[i+1:])
    return sorted(candidates_without_letter)

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
    n = len(word)
    candidates_with_letter = []
    for a in alphabet:
        for i in range(n + 1):
            candidates_with_letter.append(word[:i] + a + word[i:])
    return sorted(candidates_with_letter)

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
    if word == "" or not alphabet:
        return []
    replaced_candidates = []
    for i in range(len(word)):
        for letter in alphabet:
            candidate = word[:i] + letter + word[i+1:]
            replaced_candidates.append(candidate)
    return sorted(replaced_candidates)

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
    if not isinstance(word, str) or len(word) < 2:
        return []
    swapped_candidates = []
    for i in range(len(word)-1):
        candidate = word[:i] + word[i+1] + word[i] + word[i+2:]
        swapped_candidates.append(candidate)
    return sorted(swapped_candidates)

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
    if word == "":
        return add_letter(word, alphabet)
    generated_candidates = []
    generated_candidates.extend(delete_letter(word))
    generated_candidates.extend(swap_adjacent(word))
    if alphabet:
        generated_candidates.extend(add_letter(word, alphabet))
        generated_candidates.extend(replace_letter(word, alphabet))
    return sorted(generated_candidates)

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
    result = set(candidates)
    for candidate in candidates:
        candidates_new = generate_candidates(candidate, alphabet)
        if candidates_new is None:
            return None
        result.update(candidates_new)
    return tuple(sorted(result))

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
    candidates = propose_candidates(word, alphabet)
    if candidates is None:
        result = {token: 1.0 for token in frequencies}
        return result
    result = dict()
    for key,value in frequencies.items():
        if key not in candidates:
            result[key] = 1.0
        else:
            result[key] = 1.0 - value
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

