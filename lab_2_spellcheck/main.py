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
    return {element: tokens.count(element) / len(tokens) for element in tokens}



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
    return  (1 - len(set(token).intersection(set(candidate))) / 
            len(set(token).union(set(candidate))))


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
    if any([
        not isinstance(first_token, str),
        not check_dict(vocabulary, str, float, False),
        method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"],
        not check_list(alphabet, str, False) 
        and alphabet is not None 
        and method == "frequency-based"
            ]):
        return None
    distance = {}
    if method == "jaccard":
        for key in vocabulary.keys():
            jaccard_distance = calculate_jaccard_distance(first_token, key)
            if jaccard_distance is None:
                return None
            distance[key] = jaccard_distance
    elif method == "frequency-based":
        if alphabet is None:
            return {key: 1.0 for key in vocabulary.keys()}
        distance = calculate_frequency_distance(first_token, vocabulary, alphabet) or None
    elif method == "levenshtein":
        for key in vocabulary.keys():
            levenshtein_distance = calculate_levenshtein_distance(first_token, key)
            if levenshtein_distance is None:
                return None
            distance[key] = levenshtein_distance
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
    if any([
        not isinstance(wrong_word, str),
        not check_dict(vocabulary, str, float, False),
        method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"],
        all([not check_list(alphabet, str, False),
            alphabet is not None,
            method in ('frequency-based', 'levenshtein')
            ])]):
        return None
    wrong_word_dict = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if not wrong_word_dict:
        return None
    min_distance = min(wrong_word_dict.values())
    min_keys = [key for key, value in wrong_word_dict.items() if value == min_distance]
    min_keys_sorted = sorted(min_keys, key = lambda word: abs(len(wrong_word) - len(word)))
    new_sorted_list = [min_keys_sorted[0]]
    if len(min_keys_sorted) > 1:
        for el in min_keys_sorted:
            index = 1
            if len(min_keys_sorted[index]) == len(min_keys_sorted[0]):
                new_sorted_list.append(el)
                index += 1
            else:
                break
        return new_sorted_list.sort()[0]
    return min_keys_sorted[0] # i love this code so much omg lol pls rate me 11


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
    matrix = [[0 for j in range(candidate_length + 1)] for i in range(token_length + 1)]
    for i in range(1, candidate_length + 1):
        matrix[0][i] = i
    for j in range(1, token_length + 1):
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
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None
    matrix = initialize_levenshtein_matrix(len(token), len(candidate))
    if matrix == None:
        return None
    for i in range(1, len(token) + 1):
        for j in range(1, len(candidate) + 1):
            if token[i - 1] == candidate[j - 1]:
                cost = 0
            else:
                cost = 1
            del_symbol = matrix[i - 1][j] + 1
            add_symbol = matrix[i][j - 1] + 1
            replace_symbol = matrix[i - 1][j - 1] + cost
            matrix[i][j] = min(del_symbol, add_symbol, replace_symbol)
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
    possible_words = []
    for i in range (len(word)):
        word_wo_letter = word[:i] + word[i + 1:]
        possible_words.append(word_wo_letter)
    return sorted(possible_words)


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
    if any([
        not isinstance(word, str),
        not check_list(alphabet, str, False)
        ]):
        return []
    possible_words = []
    for i in range (len(word) + 1):
        for letter in alphabet:
            word_with_added_letter = word[:i] + letter + word[i:]
            possible_words.append(word_with_added_letter)
    return sorted(possible_words)


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
    if any([
        not isinstance(word, str),
        not check_list(alphabet, str, False)
        ]):
        return []
    possible_words = []
    for i in range (len(word)):
        for letter in alphabet:
            word_with_added_letter = word[:i] + letter + word[i + 1:]
            possible_words.append(word_with_added_letter)
    return sorted(possible_words)


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
    possible_words = []
    for i in range (len(word) - 1):
        word_wo_letter = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        possible_words.append(word_wo_letter)
    return sorted(possible_words)


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
    if any([
        not isinstance(word, str),
        not check_list(alphabet, str, True)
    ]):
        return None
    candidates_list = (delete_letter(word) + add_letter(word, alphabet) +
                       replace_letter(word, alphabet) + swap_adjacent(word))
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
    if any([
        not isinstance(word, str),
        not check_list(alphabet, str, True)
    ]):
        return None
    first_gen_candidates = generate_candidates(word, alphabet)
    if not check_list(first_gen_candidates, str, True) or first_gen_candidates is None:
        return None
    all_second_gen_candidates =  []
    for element in first_gen_candidates:
        second_gen_candidates = generate_candidates(element, alphabet)
        if second_gen_candidates is None:
            return None
        all_second_gen_candidates.append(second_gen_candidates)
    second_gen_candidates = [element for list in all_second_gen_candidates for element in list]
    first_gen_candidates += second_gen_candidates
    return tuple(sorted(list(set(first_gen_candidates)))) 
    # omg i don't love this code at all but pls rate me 11


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
    if any([
        not isinstance(word, str),
        not check_dict(frequencies, str, float, False),
        not check_list(alphabet, str, True),
    ]):
        return None
    candidates = propose_candidates(word, alphabet)
    suitable_candidates = set()
    if candidates:
        suitable_candidates = set(frequencies.keys()).intersection(set(candidates))
    probable_corrected_words = {key: 1.0 for key in frequencies.keys()}
    for candidate in suitable_candidates:
        if candidate in frequencies:
            probable_corrected_words[candidate] = 1.0 - frequencies[candidate]
        else:
            probable_corrected_words[candidate] = 1.0
    return probable_corrected_words #lmao


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
