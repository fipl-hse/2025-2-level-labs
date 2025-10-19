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
    tokenized = {}
    all_words = len(tokens)
    for word in tokens:
        tokenized[word] = tokenized.get(word, 0) + 1
    for word in tokenized:
        tokenized[word] /= all_words
    return tokenized


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
    bad_words = []
    for token in tokens:
        if token not in vocabulary:
            bad_words.append(token)
    return bad_words


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
    tokenized1 = set(token)
    tokenized2 = set(candidate)
    if not tokenized1 and not tokenized2:
        return 1.0
    intersected = tokenized1.intersection(tokenized2)
    united = tokenized1.union(tokenized2)
    jaccard_dictance = 1 - len(intersected) / len(united)
    return jaccard_dictance


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
        or (alphabet is not None and not check_list(alphabet, str, False))
    ):
        return None
    result: dict[str, float] = {}
    if method == "frequency-based":
        freq_dict = calculate_frequency_distance(first_token, vocabulary, alphabet or [])
        if freq_dict is None:
            return None
        return freq_dict
    for vocab_word in vocabulary:
        if method == "levenshtein":
            distance = calculate_levenshtein_distance(first_token, vocab_word)
        elif method == "jaccard":
            distance = calculate_jaccard_distance(first_token, vocab_word)
        else:
            return None
        if distance is None:
            return None
        result[vocab_word] = distance
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
    if (
        not isinstance(wrong_word, str)
        or not check_dict(vocabulary, str, float, False)
        or method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]
        or (alphabet is not None and not check_list(alphabet, str, False))
    ):
        return None
    distances = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if not distances or distances is None:
        return None
    min_dist_value = float("inf")
    for value in distances.values():
        min_dist_value = min(min_dist_value, value)
    closest_candidates = []
    for word, dist in distances.items():
        if dist == min_dist_value:
            closest_candidates.append(word)
    if len(closest_candidates) == 1:
        return closest_candidates[0]
    min_len_diff = float("inf")
    for word in closest_candidates:
        diff = abs(len(word) - len(wrong_word))
        if diff < min_len_diff:
            min_len_diff = diff
    length_filtered = []
    for word in closest_candidates:
        if abs(len(word) - len(wrong_word)) == min_len_diff:
            length_filtered.append(word)
    return min(length_filtered)


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
    for token_index in range(token_length + 1):
        row = []
        for cand_index in range(candidate_length + 1):
            row.append(0)
        matrix.append(row)
    for token_index in range(token_length + 1):
        matrix[token_index][0] = token_index
    for cand_index in range(candidate_length + 1):
        matrix[0][cand_index] = cand_index
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
    if token is None or candidate is None:
        return None
    m = len(token)
    n = len(candidate)
    matrix = initialize_levenshtein_matrix(m, n)
    if matrix is None:
        return None
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if token[i - 1] == candidate[j - 1]:
                cost = 0
            else:
                cost = 1
            deletion = matrix[i - 1][j] + 1
            insertion = matrix[i][j - 1] + 1
            substitution = matrix[i - 1][j - 1] + cost
            if deletion <= insertion and deletion <= substitution:
                matrix[i][j] = deletion
            elif insertion <= deletion and insertion <= substitution:
                matrix[i][j] = insertion
            else:
                matrix[i][j] = substitution
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
    if token is None or candidate is None:
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
    if not isinstance(word, str) or not word:
        return []
    result = []
    for i in range(len(word)):
        new_word = word[:i] + word[i+1:]
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
    result = []
    for i in range(len(word) + 1):
        for letter in alphabet:
            new_word = word[:i] + letter + word[i:]
            result.append(new_word)
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
    result = []
    for i, current_letter in enumerate(word):
        for letter in alphabet:
            if current_letter != letter:
                new_word = word[:i] + letter + word[i + 1:]
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
    if not isinstance(word, str) or len(word) < 2:
        return []
    result = []
    for i in range(len(word) - 1):
        swapped = list(word)
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        result.append("".join(swapped))
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
    candidates = []
    candidates.extend(delete_letter(word))
    candidates.extend(add_letter(word, alphabet))
    candidates.extend(replace_letter(word, alphabet))
    candidates.extend(swap_adjacent(word))
    return sorted(candidates)


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
    if word == "" and not alphabet:
        return ()
    candidates_set = {word}
    first_pass = generate_candidates(word, alphabet)
    if first_pass is None:
        return None
    candidates_set.update(first_pass)
    for base in first_pass:
        second_pass = generate_candidates(base, alphabet)
        if second_pass is None:
            return None
        candidates_set.update(second_pass)
    return tuple(sorted(candidates_set))


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
    freq_distance = {token: 1.0 for token in frequencies}
    if word == "":
        return freq_distance
    candidates = propose_candidates(word, alphabet)
    if candidates is None:
        return freq_distance
    for candidate in candidates:
        if candidate in frequencies:
            freq_distance[candidate] = 1.0 - frequencies[candidate]
    return freq_distance



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
