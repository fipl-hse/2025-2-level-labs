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

    freq_tokens = {}
    tokens_total = len(tokens)

    for token in tokens:
        freq_tokens[token] = freq_tokens.get(token, 0) +1


    return {token: count / tokens_total for token, count in freq_tokens.items()}

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

    incor_words = []
    for token in tokens:
        if not token in vocabulary:
            incor_words.append(token)

    return incor_words


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

    set_1 = set(token)
    set_2 = set(candidate)

    set_intersection = set_1.intersection(set_2)
    set_union = set_1.union(set_2)

    if not set_union:
        return 1.0

    jac_coef = len(set_intersection) / len(set_union)
    jac_dis = 1 - jac_coef

    return jac_dis


def calculate_distance(
    first_token: str,
    vocabulary: dict[str, float],
    method: Literal["jaccard", "frequency-based", "levenshtein"],
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
    if not isinstance(method, str) or method not in ["jaccard", "frequency-based", "levenshtein"]:
        return None
    if alphabet is not None and not check_list(alphabet, str, False):
        return None

    if method == "frequency-based":
        if alphabet is None:
            return {word: 1.0 for word in vocabulary}
        result = calculate_frequency_distance(first_token, vocabulary, alphabet)
        return result if result is not None else None

    distance_dict = {}
    if method == "jaccard":
        distance_func = calculate_jaccard_distance
    elif method == "levenshtein":
        distance_func = calculate_levenshtein_distance
    else:
        return None

    for word in vocabulary:
        dist_value = distance_func(first_token, word)
        if dist_value is None:
            return None
        distance_dict[word] = dist_value

    return distance_dict

def find_correct_word(
    wrong_word: str,
    vocabulary: dict[str, float],
    method: Literal["jaccard", "frequency-based", "levenshtein"],
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

    if alphabet is not None:
        if not check_list(alphabet, str, False):
            return None

    if method not in ["jaccard", "frequency-based", "levenshtein"]:
        return None

    dist_dict = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if not dist_dict:
        return None

    dist_min = min(dist_dict.values())

    min_dist_words = [word for word, dist in dist_dict.items() if dist == dist_min]
    if len(min_dist_words) == 1:
        return min_dist_words[0]

    length_diffs = [abs(len(word) - len(wrong_word)) for word in min_dist_words]
    min_length_diff = min(length_diffs)

    length_candidates = [min_dist_words[i] for i in range(len(min_dist_words))
                        if length_diffs[i] == min_length_diff]


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
    if not isinstance(token_length, int) or token_length < 0:
        return None
    if not isinstance(candidate_length, int) or candidate_length < 0:
        return None

    matrix = [[0] * (candidate_length + 1) for _ in range(token_length + 1)]

    for j in range(candidate_length + 1):
        matrix[0][j] = j

    for i in range(token_length + 1):
        matrix[i][0] = i

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
    if not isinstance(token, str):
        return None
    if not isinstance(candidate, str):
        return None

    token_length = len(token)
    candidate_length = len(candidate)
    matrix = initialize_levenshtein_matrix(token_length, candidate_length)

    if matrix is None:
        return None

    for i in range(1, token_length + 1):
        for j in range(1, candidate_length + 1):
            if token[i-1] == candidate[j-1]:
                cost = 0
            else:
                cost = 1

            deletion = matrix[i-1][j] + 1
            insertion = matrix[i][j-1] + 1
            substitution = matrix[i-1][j-1] + cost

            matrix[i][j] = min(deletion, insertion, substitution)

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
    if not isinstance(token, str):
        return None
    if not isinstance(candidate, str):
        return None

    matrix = fill_levenshtein_matrix(token, candidate)

    if matrix is None:
        return None

    token_len = len(token)
    candidate_len = len(candidate)

    return matrix[token_len][candidate_len]


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
    if not word:
        return []

    result = []
    for i in range(len(word)):
        possible_word = word[:i] + word[i+1:]
        result.append(possible_word)

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
    if not isinstance(word, str):
        return []
    if not check_list(alphabet, str, False):
        return []

    result = []

    for i in range(len(word) + 1):
        for letter in alphabet:
            candidate = word[:i] + letter + word[i:]
            result.append(candidate)

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
    if not isinstance(word, str):
        return []
    if not check_list(alphabet, str, False):
        return []
    if not all(isinstance(syb, str) for syb in alphabet):
        return []

    result = []

    for i, current_letter in enumerate(word):
        for letter in alphabet:
            if letter != current_letter:
                possible_word = word[:i] + letter + word[i+1:]
                result.append(possible_word)
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

    for i in range(len(word) -1):
        possible_word = word[:i] + word[i+1] + word[i] + word[i+2:]
        result.append(possible_word)

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
    if not isinstance(word, str):
        return None
    if not check_list(alphabet, str, True):
        return None

    if word == "":
        return sorted(alphabet)

    generated_candidates = []

    delete_candidates = delete_letter(word)
    if delete_candidates is not None:
        generated_candidates.extend(delete_candidates)

    swap_candidates = swap_adjacent(word)
    if swap_candidates is not None:
        generated_candidates.extend(swap_candidates)

    add_candidates = add_letter(word, alphabet)
    if add_candidates is not None:
        generated_candidates.extend(add_candidates)

    replace_candidates = replace_letter(word, alphabet)
    if replace_candidates is not None:
        generated_candidates.extend(replace_candidates)

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
    if not isinstance(word, str) or not check_list(alphabet, str, True):
        return None

    if word == "" and len(alphabet) == 0:
        return ()

    candidate_set = {word}

    level1_candidates = generate_candidates(word, alphabet)
    if not level1_candidates:
        return None
    candidate_set.update(level1_candidates)

    for candidate in level1_candidates:
        level2_candidates = generate_candidates(candidate, alphabet)
        if not level2_candidates:
            return None
        candidate_set.update(level2_candidates)

    return tuple(sorted(candidate_set))


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
    if (
        not isinstance(word, str)
        or not check_dict(frequencies, str, float, False)
        or not check_list(alphabet, str, True)
):
        return None

    if not frequencies:
        return {}

    candidates = propose_candidates(word, alphabet)
    frequency_distances = {token: 1.0 for token in frequencies.keys()}
    if candidates is None or not candidates:
        return frequency_distances


    for candidate in set(candidates) & set(frequencies.keys()):
        if candidate in frequencies:
            frequency_distances[candidate] = 1.0 - frequencies[candidate]

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
