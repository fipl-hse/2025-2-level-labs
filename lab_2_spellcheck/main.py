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
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        return None
    
    total = len(tokens)
    vocab: dict[str, float] = {}

    for token in tokens:
        vocab[token] = vocab.get(token, 0) + 1

    for token in vocab:
        vocab[token] /= total
    
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
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        return None
    if not isinstance(vocabulary, dict) or not all(isinstance(k, str) for k in vocabulary):
        return None
    
    nonvocab = [token for token in tokens if token not in vocabulary]
    return nonvocab


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
        return 0.0
    
    set_token = set(token)
    set_candidate = set(candidate)
    intersection = set_token & set_candidate
    union = set_token | set_candidate

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
    if not isinstance(first_token, str):
        return None
    if not isinstance(vocabulary, dict) or not all(isinstance(k,str) for k in vocabulary):
        return None
    if method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]:
        return None
    
    distances: dict[str, float] = {}

    for word in vocabulary:
        if method == "jaccard":
            dist = calculate_jaccard_distance(first_token, word)
            if dist is None:
                return None
            distances[word] = dist
        elif method == "frequency-based":
            freq = vocabulary[word]
            distances[word] = 1 - freq
        else:
            distances[word] = 0.0

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
    if not isinstance(wrong_word, str):
        return None
    if not isinstance(vocabulary, dict) or len(vocabulary) == 0:
        return None
    
    distances = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if distances is None:
        return None
    
    min_distance = min(distances.values())
    candidates = [word for word, dist in distances.items() if dist == min_distance]

    candidates.sort(key=lambda w: (abs(len(w) - len(wrong_word)), w))
    return candidates[0]


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
    if token is None or candidate is None:
        return None
    m, n = len(token), len(candidate)
    matrix = initialize_levenshtein_matrix(m,n)
    if matrix is None:
        return None
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if token[i-1] == candidate[j-1]:
                cost = 0
            else:
                cost = 1

            matrix[i][j] = min(
                matrix[i-1][j] + 1,
                matrix[i][j-1] + 1,
                matrix[i-1][j-1] + cost
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
    if token is None or candidate is None:
        return None
    
    matrix = fill_levenshtein_matrix(token,candidate)
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
    if not(isinstance(word,str)):
        return []
    
    variant = [word[:i] + word[i+1:] for i in range(len(word))]

    return sorted(variant)


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

if __name__ == "__main__":
    tokens = ['проверка', 'код', 'пишу', 'проверка', 'кода']

    vocab = build_vocabulary(tokens)
    print("Vocabalary:", vocab)

    test_tokens = ['проверка', 'код', 'такой']
    nonvocab = find_out_of_vocab_words(test_tokens, vocab)
    print("Not in vocab:", nonvocab)

    word1 = "код"
    word2 = "кода"
    jaccard_dist= calculate_jaccard_distance(word1,word2)
    print(f"jaccard distance between '{word1}' and '{word2}':", jaccard_dist)

    jaccard_scores = calculate_distance(word1, vocab, method = "jaccard")
    print(f"jaccard dist '{word1}':", jaccard_scores)

    freq_score = calculate_distance(word1, vocab, method="frequency-based")
    print(f"Frequency-based '{word1}':", freq_score)

    wrong_word = "кд"
    corrected = find_correct_word(wrong_word, vocab, method="jaccard")
    print(f"correct word '{wrong_word}':", corrected)

    tests_levenshtein = [
        ("код", "код", 0),
        ("проверка", "праверка", 1),
        ("такой", "тако", 1),
        ("пишу", "пишу", 0)
    ]

    for a,b, expected in tests_levenshtein:
        result = calculate_levenshtein_distance(a,b)
        print(f"Levenshtein('{a}','{b}') = {result} (expected {expected})")

    word = "код"
    variant = delete_letter(word)
    print(f"variant w/delete letter '{word}':", variant)