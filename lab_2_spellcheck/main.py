"""
Lab 2.
"""

# pylint:disable=unused-argument
from typing import Literal, cast
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
    vocab: dict[str, float] = {}
    total = len(tokens)
    for token in tokens:
        if not isinstance(token, str):
            return None
        vocab[token] = vocab.get(token, 0) + 1 / total
    return vocab


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
    for t in tokens:
        if not isinstance(t, str):
            return None
    if not check_dict(vocabulary, str, float, False):
        return None
    for k, v in vocabulary.items():
        if not isinstance(k, str) or not isinstance(v, float):
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
    tokens = set(token)
    candidates = set(candidate)
    union = tokens | candidates
    intersection = tokens & candidates
    if len(union) == 0 or len(token) == 0 or len(candidate) == 0:
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
    if (
        not isinstance(first_token, str)
        or not check_dict(vocabulary, str, float, False)
        or any(not isinstance(k, str)
        or not isinstance(v, (int, float)) for k, v in vocabulary.items())
        or method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]
        ):
        return None
    if (alphabet is not None and not check_list(alphabet, str, False)):
        return None
    if method == "frequency-based":
        if alphabet is None:
            return {word: 1.0 for word in vocabulary}
        distances = calculate_frequency_distance(first_token, vocabulary, alphabet)
        if distances is None:
            return None
        return distances
    method_funcs = {
        "jaccard": calculate_jaccard_distance,
        "levenshtein": calculate_levenshtein_distance,
        "jaro-winkler": calculate_jaro_winkler_distance,
    }
    func = method_funcs[method]
    result: dict[str, float] = {}
    for word in vocabulary:
        dist = func(first_token, word)
        if dist is None:
            return None
        result[word] = float(dist)
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
    if (not isinstance(wrong_word, str)
    or method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]
    or not check_dict(vocabulary, str, float, False)
    or not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in vocabulary.items())):
        return None
    if alphabet is not None:
        if not check_list(alphabet, str, False) or not all(isinstance(ch, str) for ch in alphabet):
            return None
    distances = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if not check_dict(distances, str, float, False):
        return None
    min_distance = min(distances.values())
    candidates = [word for word, dist in distances.items() if dist == min_distance]
    if not candidates:
        return None
    wrong_len = len(wrong_word)
    min_len_diff = min(abs(len(c) - wrong_len) for c in candidates)
    filtered = [c for c in candidates if abs(len(c) - wrong_len) == min_len_diff]
    return sorted(filtered)[0] if filtered else None


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
    if not isinstance(token_length, int):
        return None
    if not isinstance(candidate_length, int):
        return None
    if token_length < 0 or candidate_length < 0:
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
            cost = 0 if token[i - 1] == candidate[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost)
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
    result: list[str] = []
    for i in range(len(word)):
        new_word = word[:i] + word[i + 1:]
        result.append(new_word)
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
    if not isinstance(word, str) or not check_list(alphabet, str, False):
        return []
    result: list[str] = []
    for i in range(len(word) + 1):
        for ch in alphabet:
            result.append(word[:i] + ch + word[i:])
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
    if not isinstance(word, str) or not check_list(alphabet, str, False):
        return []
    result: list[str] = []
    for i in range(len(word)):
        for ch in alphabet:
            new_word = word[:i] + ch + word[i + 1:]
            result.append(new_word)
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
    result: list[str] = []
    for i in range(len(word) - 1):
        new_word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        result.append(new_word)
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
    if not isinstance(word, str) or not check_list(alphabet, str, True):
        return None
    result: list[str] = []
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
    if not isinstance(word, str) or not check_list(alphabet, str, True):
        return None
    primary = generate_candidates(word, alphabet)
    if primary is None:
        return None
    all_candidates: set[str] = set(primary)
    for candidate in primary:
        secondary = generate_candidates(candidate, alphabet)
        if secondary is None:
            return None
        all_candidates.update(secondary)
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
    if (not isinstance(word, str)
        or not check_dict(frequencies, str, float, False)
        or not check_list(alphabet, str, True)):
        return None
    candidates = propose_candidates(word, alphabet)
    if not candidates:
        distance_cand: dict[str, float] = {}
        for token, freq in frequencies.items():
            distance_cand[token] = float(freq) if token == word else 1.0
        return distance_cand
    distance: dict[str, float] = {}
    for token in frequencies:
        if token in candidates:
            distance[token] = 1.0 - float(frequencies[token])
        else:
            distance[token] = 1.0
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
    if (not isinstance(token, str)
    or not isinstance(candidate, str)
    or not isinstance(match_distance, int)
    or match_distance < 0):
        return None
    candidate_len = len(candidate)
    token_matches = [False for _ in token]
    candidate_matches = [False for _ in candidate]
    matches = 0
    max_distance = max(match_distance, 0)
    for i, t_char in enumerate(token):
        start = max(0, i - max_distance)
        end = min(i + max_distance + 1, candidate_len)
        for j in range(start, end):
            if not candidate_matches[j] and t_char == candidate[j]:
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
    if not isinstance(token, str):
        return None
    if not isinstance(candidate, str):
        return None
    if not check_list(token_matches, bool, True):
        return None
    if not check_list(candidate_matches, bool, True):
        return None
    if len(token) != len(token_matches):
        return None
    if len(candidate) != len(candidate_matches):
        return None
    transpositions = 0
    candidate_index = 0
    for i, matched in enumerate(token_matches):
        if matched:
            while candidate_index < len(candidate) and not candidate_matches[candidate_index]:
                candidate_index += 1
            if candidate_index < len(candidate) and token[i] != candidate[candidate_index]:
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
    if not isinstance(matches, int) or matches < 0:
        return None
    if not isinstance(transpositions, int) or transpositions < 0:
        return None
    if len(token) == 0 or len(candidate) == 0:
        return None
    if matches == 0:
        return 1.0
    len_token = len(token)
    len_candidate = len(candidate)
    jaro = (
        matches / len_token
        + matches / len_candidate
        + (matches - transpositions) / matches
    ) / 3
    return 1.0 - jaro


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
    if (not isinstance(token, str)
        or not isinstance(candidate, str)
        or not isinstance(jaro_distance, float)
        or not isinstance(prefix_scaling, float)):
        return None
    prefix_len = 0
    max_prefix = 4
    for i in range(min(len(token), len(candidate), max_prefix)):
        if token[i] != candidate[i]:
            break
        prefix_len += 1
    bonus = jaro_distance * prefix_len * prefix_scaling
    return float(bonus)


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
    if (not isinstance(token, str)
        or not isinstance(candidate, str)
        or not isinstance(prefix_scaling, (float, int))):
        return None
    if token == "" and candidate == "":
        return 1.0
    if token == "" or candidate == "":
        return 1.0
    match_distance = max(len(token), len(candidate)) // 2 - 1
    match_distance = max(match_distance, 0)
    matches_result = get_matches(token, candidate, match_distance)
    if matches_result is None:
        return None
    matches_count, token_matches, candidate_matches = matches_result
    if matches_count == 0:
        return 1.0
    transpositions_count = count_transpositions(token, candidate, token_matches, candidate_matches)
    if transpositions_count is None:
        return None
    jaro_distance = calculate_jaro_distance(token, candidate, matches_count, transpositions_count)
    if jaro_distance is None:
        return None
    winkler_bonus = winkler_adjustment(token, candidate, jaro_distance, prefix_scaling)
    if winkler_bonus is None:
        return None
    jaro_similarity = 1.0 - jaro_distance
    jaro_winkler_similarity = jaro_similarity + winkler_bonus
    return float(1.0 - jaro_winkler_similarity)
