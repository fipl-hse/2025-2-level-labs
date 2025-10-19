"""
Lab 2.
"""

# pylint:disable=unused-argument
from typing import Literal
from typing import Any

def check_list(user_input: Any, elements_type: type, can_be_empty: bool) -> bool:
    """
    Check if the object is a list containing elements of a certain type.

    Args:
        user_input (Any): Object to check
        elements_type (type): Expected type of list elements
        can_be_empty (bool): Whether an empty list is allowed

    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(user_input, list):
        return False
    if not user_input:
        return can_be_empty
    return all(isinstance(element, elements_type) for element in user_input)

def check_dict(user_input: Any, key_type: type, value_type: type, can_be_empty: bool) -> bool:
    """
    Check if the object is a dictionary with keys and values of given types.

    Args:
        user_input (Any): Object to check
        key_type (type): Expected type of dictionary keys
        value_type (type): Expected type of dictionary values
        can_be_empty (bool): Whether an empty dictionary is allowed

    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(user_input, dict):
        return False
    if not user_input:
        return can_be_empty
    return (all(isinstance(key, key_type) for key in user_input) and
        all(isinstance(value, value_type) for value in user_input.values()))

def clean_and_tokenize(text: str) -> list[str] | None:
    """
    Remove punctuation, convert to lowercase, and split into tokens.

    Args:
        text (str): Original text

    Returns:
        list[str] | None: A list of lowercase tokens without punctuation.
        In case of corrupt input arguments, None is returned.
    """
    if not isinstance(text, str):
        return None
    words = text.lower().split()
    tokens = []
    for word in words:
        cleaned_word = ''.join(
            symbol for symbol in word
            if symbol.isalnum())
        if cleaned_word:
            tokens.append(cleaned_word)
    return tokens

def remove_stop_words(tokens: list[str], stop_words: list[str]) -> list[str] | None:
    """
    Exclude stop words from the token sequence.

    Args:
        tokens (list[str]): Original token sequence
        stop_words (list[str]): Tokens to exclude

    Returns:
        list[str] | None: Token sequence without stop words.
        In case of corrupt input arguments, None is returned.
    """
    if not all([check_list(tokens, str, True),
        check_list(stop_words, str, True)]):
        return None
    return [token for token in tokens if token not in set(stop_words)]

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

    unique_tokens = set(tokens)
    all_tokens_length = len(tokens)

    return {token : tokens.count(token) / all_tokens_length for token in unique_tokens}


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

    tok = set(token)
    cand = set(candidate)
    return 1 - len(tok & cand)/len(tok | cand)


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
    if (not vocabulary or
    (alphabet is not None and not check_list(alphabet, str, True))):
        return None

    if method == "frequency-based":
        if alphabet is None:
            return {token: 1.0 for token in vocabulary}
        fr_dist = calculate_frequency_distance(first_token, vocabulary, alphabet)
        if fr_dist is None:
            return None
        res = fr_dist

    #add other methods here

    return res


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
    if not (
        isinstance(wrong_word, str) and
        check_dict(vocabulary, str, float, False) and
        method in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"] and
        (alphabet is None or check_list(alphabet, str, False))
    ):
        return None
    distances = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if not distances:
        return None

    min_value = min(distances.values())
    candidates = [candidate for candidate, value in distances.items() if value == min_value]
    if not candidates:
        return None

    if len(candidates) > 1:
        min_length_diff = min(abs(len(candidate) - len(wrong_word)) for candidate in candidates)
        closest_candidates = [candidate for candidate in candidates
                              if abs(len(candidate) - len(wrong_word)) == min_length_diff]
        return sorted(closest_candidates)[0]

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

    shortened_words = []
    for i in range(len(word)):
        new_word = word[:i] + word[i+1:]
        shortened_words.append(new_word)
    return sorted(shortened_words)


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
    extended_words = []
    for i in range(len(word) + 1):
        for letter in alphabet:
            new_word = word[:i] + letter + word[i:]
            extended_words.append(new_word)
    return sorted(extended_words)


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
    if not isinstance(word, str) or not check_list(alphabet, str, False) or word is None:
        return []

    replased_letters_words = []
    for i in range(len(word)):
        for letter in alphabet:
            new_word = word[:i] + letter + word[i + 1:]
            replased_letters_words.append(new_word)
    return sorted(replased_letters_words)


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

    replaced_letters_words = []
    for i in range(len(word) - 1):
        new_word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        replaced_letters_words.append(new_word)
    return sorted(replaced_letters_words)


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
    