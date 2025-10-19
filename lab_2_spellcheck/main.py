"""
Lab 2.
"""

# pylint:disable=unused-argument
from typing import Literal

from lab_1_keywords_tfidf.main import check_dict, check_float, check_list, check_positive_int


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
    return {token: tokens.count(token) / len(tokens) for token in tokens}


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
    if not all([check_dict(vocabulary, str, float, False),
                check_list(tokens, str, False)]):
        return None
    return [token for token in tokens if token not in vocabulary.keys()]


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
    if not token and not candidate:
        return 1.0
    if not (isinstance(token, str) and isinstance(candidate, str)):
        return None
    token_scores = set(token)
    word_scores = set(candidate)
    jaccard_distance = 1 - len(
        token_scores.intersection(word_scores)) / len(
        token_scores.union(word_scores))
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
    if (not check_dict(vocabulary, str, float, False) or
        not isinstance(first_token, str) or
        not isinstance(method, str) or
        not method in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]
    ):
        return None
    if method == "frequency-based":
        if not alphabet:
            return dict.fromkeys(vocabulary, 1.0)
        return calculate_frequency_distance(first_token, vocabulary, alphabet)

    distances = {}

    if method == "jaccard":
        for candidate in vocabulary:
            distance = calculate_jaccard_distance(first_token, candidate)
            if distance is None:
                return None
            distances[candidate] = distance
        return distances
    if method == "levenshtein":
        for candidate in vocabulary:
            distance = calculate_levenshtein_distance(first_token, candidate)
            if distance is None:
                return None
            distances[candidate] = distance
        return distances
    if method == "jaro-winkler":
        for candidate in vocabulary:
            distance = calculate_jaro_winkler_distance(first_token, candidate)
            if distance is None:
                return None
            distances[candidate] = distance
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
    if (
        not isinstance(wrong_word, str)
        or not check_dict(vocabulary, str, float, False)
        or not method in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]
        or not (alphabet is None or check_list(alphabet, str, True))
    ):
        return None

    distances = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if distances is None:
        return None
    min_value = min(distances.values())

    top_words = [word for word, value in distances.items() if value == min_value]

    if len(top_words) > 1:
        top_words.sort(key=lambda word: (abs(len(word) - len(wrong_word)), word))

    return top_words[0]



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

    levenshtein_matrix = []
    levenshtein_matrix.append(list(range(candidate_length + 1)))

    for i in range(1, token_length + 1):
        levenshtein_matrix_line = [i if ii == 0 else 0 for ii in range(candidate_length + 1)]
        levenshtein_matrix.append(levenshtein_matrix_line)
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
    if (not isinstance(token, str) or
        not isinstance(candidate, str)):
        return None
    levenshtein_matrix = initialize_levenshtein_matrix(len(token), len(candidate))
    if levenshtein_matrix is None:
        return None
    for i in range(1, len(token) + 1):
        for j in range(1, len(candidate) + 1):
            if token[i - 1] == candidate[j - 1]:
                levenshtein_matrix[i][j] = levenshtein_matrix[i - 1][j - 1]
            else:
                delete_c = levenshtein_matrix[i - 1][j] + 1
                insert_c = levenshtein_matrix[i][j - 1] + 1
                replace_c = levenshtein_matrix[i - 1][j - 1] + 1
                levenshtein_matrix[i][j] = min(
                    delete_c, insert_c, replace_c)
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
    if (not isinstance(token, str) or
        not isinstance(candidate, str)):
        return None
    levenshtein_matrix = fill_levenshtein_matrix(token, candidate)
    if levenshtein_matrix is None:
        return None
    return levenshtein_matrix[-1][-1]


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
    if not all([isinstance(word, str),
                check_list(alphabet, str, False)]):
        return []
    add_letter_list = []
    for i in range(len(word) + 1):
        for char in alphabet:
            add_letter_list.append(word[:i] + char + word[i:])
    return sorted(add_letter_list)


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
    if not all([isinstance(word, str),
                check_list(alphabet, str, False)]):
        return []
    replaced_letter_list = []
    for i in range(len(word)):
        for char in alphabet:
            replaced_letter_list.append(word[:i] + char + word[i + 1:])
    return sorted(replaced_letter_list)


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
    replace_letter_list = []
    for i in range(len(word) - 1):
        replace_letter_list.append(word[:i] + word[i + 1] + word[i] + word[i + 2:])
    return sorted(replace_letter_list)


def generate_candidates(word: str, alphabet: list[str]) -> list[str] | None:
    """
    Generate all possible word words for a given word using
    four basic operations.

    Args:
        candidate (str): The input word.
        alphabet (list[str]): Alphabet for candidates creation.

    Returns:
        list[str] | None: A combined list of candidate words generated by all operations.

    In case of corrupt input arguments, None is returned.
    """
    if not all([isinstance(word, str),
                check_list(alphabet, str, True)]):
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
    if not all([isinstance(word, str), check_list(alphabet, str, True)]):
        return None
    top_words = generate_candidates(word, alphabet)
    if top_words is None:
        return None
    all_top_words = set(top_words)
    for top_word in list(all_top_words):
        second_level_top_words = generate_candidates(top_word, alphabet)
        if second_level_top_words is None:
            return None
        all_top_words.update(second_level_top_words)
    return tuple(sorted(set(all_top_words)))


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
    if not all([isinstance(word, str),
        check_dict(frequencies, str, float, False),
        check_list(alphabet, str, True)]):
        return None

    frequency_distances: dict = {token: 1.0 for token in frequencies}
    top_words = propose_candidates(word, alphabet)
    if top_words is None:
        return frequency_distances

    for top_word in top_words:
        if top_word in frequencies:
            frequency_distances[top_word] = 1.0 - frequencies[top_word]

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
    if not (
        isinstance(token, str)
        and isinstance(candidate, str)
        and isinstance(match_distance, int)
        and check_positive_int(match_distance)
        or match_distance == 0
    ):
        return None

    total_matches_counter = 0
    token_matches = [False] * len(token)
    candidate_matches = [False] * len(candidate)

    for token_index, token_char in enumerate(token):
        start = max(0, token_index - match_distance)
        end = token_index + match_distance + 1
        candidate_slice = candidate[start:end]
        for offset, candidate_char in enumerate(candidate_slice):
            candidate_index = start + offset
            if candidate_char == token_char and not candidate_matches[candidate_index]:
                total_matches_counter += 1
                token_matches[token_index] = True
                candidate_matches[candidate_index] = True
                break
    return total_matches_counter, token_matches, candidate_matches


def count_transpositions(
    token: str, candidate: str, token_matches: list[bool], candidate_matches: list[bool]
) -> int | None:
    """
    Count the number of transpositions between two strings based on matching letters.

    Args:
        token (str): The first string to compare.
        candidate (str): The second string to compare.
        token_matches (list[bool]): Boolean list indicating matches in token.
        candidate_matches (list[bool]): Boolean list indicating matches in word.

    Returns:
        int | None: Number of transpositions.

    In case of corrupt input arguments, None is returned.
    """
    if not (
        isinstance(token, str)
        and isinstance(candidate, str)
        and check_list(token_matches, bool, False)
        and check_list(candidate_matches, bool, False)
    ):
        return None

    token_matches_chars = [char for char, matched in zip(token, token_matches) if matched]
    candidate_matches_chars = [
        char for char, matched in zip(candidate, candidate_matches) if matched
    ]

    transpositions_counter = sum(1
        for token_char, candidate_char in zip(token_matches_chars, candidate_matches_chars)
        if token_char != candidate_char)
    return transpositions_counter // 2


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
    if not (
        isinstance(token, str)
        and isinstance(candidate, str)
        and (check_positive_int(matches) or matches == 0)
        and (check_positive_int(transpositions) or transpositions == 0)
    ):
        return None

    if not matches:
        return 1.0
    return (1
        - (matches / len(token) + matches / len(candidate) +
           (matches - transpositions) / matches)/ 3)


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
    if not (
        isinstance(token, str)
        and isinstance(candidate, str)
        and check_float(jaro_distance)
        and check_float(prefix_scaling)
    ):
        return None

    prefix_length = 0
    for token_char, candidate_char in tuple(zip(token, candidate))[:4]:
        if token_char != candidate_char:
            break
        prefix_length += 1
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
    if not (isinstance(token, str) and isinstance(candidate, str) and check_float(prefix_scaling)):
        return None

    if not token or not candidate:
        return 1.0

    match_distance = max(len(token), len(candidate)) // 2 - 1
    match_distance = max(match_distance, 0)
    matches = get_matches(token, candidate, match_distance)
    if matches is not None:
        total_matches, token_matches, candidate_matches = matches
        if total_matches == 0:
            return 1.0
        transpositions = count_transpositions(token, candidate, token_matches, candidate_matches)
        if transpositions is not None:
            jaro_distance = calculate_jaro_distance(token, candidate, total_matches, transpositions)
            if jaro_distance is not None:
                adjustment = winkler_adjustment(token, candidate, jaro_distance)
                if adjustment is not None:
                    return jaro_distance - adjustment
    return None
