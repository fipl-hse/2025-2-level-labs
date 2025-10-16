"""
Lab 2.
"""

# pylint:disable=unused-argument
from typing import Literal


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
    if not isinstance(tokens, list):
        return None

    if not all(isinstance(token, str) for token in tokens):
        return None

    if not tokens:
        return None

    frequency_dict = {}
    total_tokens = len(tokens)

    for token in tokens:
        if token in frequency_dict:
            frequency_dict[token] += 1
        else:
            frequency_dict[token] = 1

    vocabulary = {}
    for token, count in frequency_dict.items():
        vocabulary[token] = count / total_tokens

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

    if not isinstance(tokens, list):
        return None

    if not tokens:
        return None

    if not all(isinstance(token, str) for token in tokens):
        return None

    if not isinstance(vocabulary, dict):
        return None

    if not vocabulary:
        return None

    if not all(
        isinstance(key, str) and isinstance(value, float) for key, value in vocabulary.items()
    ):
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

    set1 = set(token)
    set2 = set(candidate)

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if not union:
        return 1.0

    jaccard_similarity = len(intersection) / len(union)

    jaccard_distance = 1.0 - jaccard_similarity

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
    if not isinstance(first_token, str) or not isinstance(vocabulary, dict):
        return None

    if not vocabulary:
        return None

    for key, value in vocabulary.items():
        if not isinstance(key, str) or not isinstance(value, (int, float)):
            return None

    if method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]:
        return None

    if method == "frequency-based" and alphabet is not None:
        if not isinstance(alphabet, list) or not all(
            isinstance(letter, str) for letter in alphabet
        ):
            return None

    result = {}

    if method == "jaccard":
        for candidate in vocabulary:
            distance = calculate_jaccard_distance(first_token, candidate)
            if distance is None:
                return None
            result[candidate] = distance

    elif method == "levenshtein":
        for candidate in vocabulary:
            distance = calculate_levenshtein_distance(first_token, candidate)
            if distance is None:
                return None
            result[candidate] = float(distance)

    elif method == "jaro-winkler":
        for candidate in vocabulary:
            distance = calculate_jaro_winkler_distance(first_token, candidate)
            if distance is None:
                return None
            result[candidate] = distance

    elif method == "frequency-based":
        if alphabet is None:
            for candidate in vocabulary:
                result[candidate] = 1.0
        else:
            freq_result = calculate_frequency_distance(first_token, vocabulary, alphabet)
            if freq_result is None:
                return None
            result = freq_result

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
    if not isinstance(wrong_word, str) or not isinstance(vocabulary, dict):
        return None

    if not vocabulary:
        return None

    for key, value in vocabulary.items():
        if not isinstance(key, str) or not isinstance(value, (int, float)):
            return None

    if method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]:
        return None

    if alphabet is not None:
        if not isinstance(alphabet, list) or not all(
            isinstance(letter, str) for letter in alphabet
        ):
            return None

    distances = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if distances is None:
        return None

    min_distance = float("inf")
    best_candidates = []

    for candidate, distance in distances.items():
        if distance < min_distance:
            min_distance = distance
            best_candidates = [candidate]
        elif distance == min_distance:
            best_candidates.append(candidate)

    if not best_candidates:
        return None

    if len(best_candidates) == 1:
        return best_candidates[0]

    best_candidates.sort(key=lambda word: (abs(len(word) - len(wrong_word)), word))
    return best_candidates[0]


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
    if not isinstance(token_length, int) or not isinstance(candidate_length, int):
        return None

    if token_length < 0 or candidate_length < 0:
        return None

    matrix = []
    for row_index in range(token_length + 1):
        row = []
        for column_index in range(candidate_length + 1):
            if row_index == 0:
                row.append(column_index)
            elif column_index == 0:
                row.append(row_index)
            else:
                row.append(0)
        matrix.append(row)

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

    for token_index in range(1, len(token) + 1):
        for candidate_index in range(1, len(candidate) + 1):
            if token[token_index - 1] == candidate[candidate_index - 1]:
                substitution_cost = 0
            else:
                substitution_cost = 1

            matrix[token_index][candidate_index] = min(
                matrix[token_index - 1][candidate_index] + 1,
                matrix[token_index][candidate_index - 1] + 1,
                matrix[token_index - 1][candidate_index - 1] + substitution_cost,
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

    result = []
    for position in range(len(word)):
        new_word = word[:position] + word[position + 1 :]
        result.append(new_word)

    result.sort()
    return result


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

    result = []
    for position in range(len(word) + 1):
        for letter in alphabet:
            new_word = word[:position] + letter + word[position:]
            result.append(new_word)

    result.sort()
    return result


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

    result = []
    for position in range(len(word)):
        for letter in alphabet:
            if letter != word[position]:
                new_word = word[:position] + letter + word[position + 1 :]
                result.append(new_word)

    result.sort()
    return result


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
    for position in range(len(word) - 1):
        new_word = word[:position] + word[position + 1] + word[position] + word[position + 2 :]
        result.append(new_word)

    result.sort()
    return result


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
    if not isinstance(word, str) or not isinstance(alphabet, list):
        return None

    if not all(isinstance(letter, str) for letter in alphabet):
        return None

    delete_candidates = delete_letter(word)
    add_candidates = add_letter(word, alphabet)
    replace_candidates = replace_letter(word, alphabet)
    swap_candidates = swap_adjacent(word)

    all_candidates = delete_candidates + add_candidates + replace_candidates + swap_candidates
    unique_candidates = list(set(all_candidates))
    unique_candidates.sort()

    return unique_candidates


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
    if not isinstance(word, str) or not isinstance(alphabet, list):
        return None

    if not all(isinstance(letter, str) for letter in alphabet):
        return None

    if word == "":
        first_level = generate_candidates(word, alphabet)
        if first_level is None:
            return None

        all_candidates = set(first_level)

        for candidate in first_level:
            second_level = generate_candidates(candidate, alphabet)
            if second_level is None:
                return None
            all_candidates.update(second_level)

        return tuple(sorted(all_candidates))

    first_level = generate_candidates(word, alphabet)
    if first_level is None:
        return None

    all_candidates = set(first_level)

    for candidate in first_level:
        second_level = generate_candidates(candidate, alphabet)
        if second_level is None:
            return None
        all_candidates.update(second_level)

    return tuple(sorted(all_candidates))


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
        or not isinstance(frequencies, dict)
        or not isinstance(alphabet, list)
    ):
        return None

    if not frequencies:
        return None

    if not all(isinstance(letter, str) for letter in alphabet):
        return None

    for key, value in frequencies.items():
        if not isinstance(key, str) or not isinstance(value, (int, float)):
            return None

    candidates = propose_candidates(word, alphabet)
    if candidates is None:
        candidates = ()

    result = {}
    for token in frequencies:
        if token in candidates:
            result[token] = float(frequencies[token])
        else:
            result[token] = 1.0

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
        not isinstance(token, str)
        or not isinstance(candidate, str)
        or not isinstance(match_distance, int)
    ):
        return None

    if match_distance < 0:
        return None

    token_matches = [False] * len(token)
    candidate_matches = [False] * len(candidate)
    matches = 0

    for token_index in range(len(token)):
        start = max(0, token_index - match_distance)
        end = min(len(candidate), token_index + match_distance + 1)

        for candidate_index in range(start, end):
            if (
                not candidate_matches[candidate_index]
                and token[token_index] == candidate[candidate_index]
            ):
                token_matches[token_index] = True
                candidate_matches[candidate_index] = True
                matches += 1
                break

    return (matches, token_matches, candidate_matches)


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
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None

    if not isinstance(token_matches, list) or not isinstance(candidate_matches, list):
        return None

    if not all(isinstance(match, bool) for match in token_matches) or not all(
        isinstance(match, bool) for match in candidate_matches
    ):
        return None

    if len(token_matches) != len(token) or len(candidate_matches) != len(candidate):
        return None

    transpositions = 0
    candidate_index = 0

    for token_index in range(len(token)):
        if token_matches[token_index]:
            while candidate_index < len(candidate) and not candidate_matches[candidate_index]:
                candidate_index += 1

            if (
                candidate_index < len(candidate)
                and token[token_index] != candidate[candidate_index]
            ):
                transpositions += 1

            candidate_index += 1

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
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None

    if not isinstance(matches, int) or not isinstance(transpositions, int):
        return None

    if matches < 0 or transpositions < 0:
        return None

    if matches == 0:
        return 1.0

    token_length = len(token)
    candidate_length = len(candidate)

    jaro_similarity = (
        matches / token_length + matches / candidate_length + (matches - transpositions) / matches
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
    max_prefix_length = min(len(token), len(candidate), 4)

    for i in range(max_prefix_length):
        if token[i] == candidate[i]:
            prefix_length += 1
        else:
            break

    winkler_adjustment_value = prefix_length * prefix_scaling * jaro_distance
    return winkler_adjustment_value


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

    if not isinstance(prefix_scaling, float):
        return None

    if token == "" and candidate == "":
        return 1.0

    match_result = get_matches(token, candidate, max(len(token), len(candidate)) // 2 - 1)
    if match_result is None:
        return None

    matches, token_matches, candidate_matches = match_result

    transpositions_count = count_transpositions(token, candidate, token_matches, candidate_matches)
    if transpositions_count is None:
        return None

    jaro_distance = calculate_jaro_distance(token, candidate, matches, transpositions_count)
    if jaro_distance is None:
        return None

    winkler_adjust = winkler_adjustment(token, candidate, jaro_distance, prefix_scaling)
    if winkler_adjust is None:
        return None

    jaro_winkler_distance = jaro_distance - winkler_adjust

    return max(0.0, min(jaro_winkler_distance, 1.0))
