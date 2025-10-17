"""
Lab 2.
"""

# pylint:disable=unused-argument
from typing import Literal

def clean_and_tokenize(text: str) -> list[str] | None:
    
    if not isinstance(text, str):
        return None
    string_punct = '«»„“‚‘—–−-….,:;!?()[]""''`'
    for syb in string_punct:
        text = text.replace(syb, '')
    new_text = text.lower()
    tokens = new_text.split()
    return tokens

def remove_stop_words(tokens: list[str], stop_words: list[str]) -> list[str] | None:

    if not isinstance(tokens, list) or not isinstance(stop_words, list):
        return None
    if not all(isinstance(el, str) for el in tokens):
        return None
    if not all(isinstance(el, str) for el in stop_words):
        return None
    result = [token for token in tokens if token not in stop_words]
    return result


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
    
    freq_tokens = {}
    tokens_total = len(tokens)

    for token in tokens:
        freq_tokens[token] = freq_tokens.get(token, 0) +1

    for token in freq_tokens:
        freq_tokens[token] = freq_tokens[token] / tokens_total

    return freq_tokens

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
    if not all(isinstance(token, str) for token in tokens):
        return None
    if not isinstance(vocabulary, dict):
        return None
    if not all(isinstance(word, str) for word in vocabulary):
        return None
    if not all(isinstance(word_freq, float) for word_freq in vocabulary.values()):
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
    if not isinstance(token, str):
        return None
    if not isinstance(candidate, str):
        return None
    if len(token) == 0 and len(candidate) == 0:
        return 1.0

    set_1 = set(token)
    set_2 = set(candidate)

    set_intersection = set_1.intersection(set_2)
    set_union = set_1.union(set_2)

    if len(set_union) == 0:
        return 0.0
    
    jac_coef = len(set_intersection) / len(set_union)
    jac_dis = 1 - jac_coef

    return jac_dis


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
    if not isinstance(vocabulary, dict):
        return None
    if not all(isinstance(word, str) for word in vocabulary.keys()):
        return None
    if not all(isinstance(word_freq, float) for word_freq in vocabulary.values()):
        return None
    if method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]:
        return None
    
    if method == "jaccard":
        dist = {}
        for word in vocabulary:
            jac_dis = calculate_jaccard_distance(first_token, word)
            if jac_dis is not None:
                dist[word] = jac_dis
    
        return dist
    
    return None #так как не реализуем другие методы?


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
    if not isinstance(vocabulary, dict):
        return None
    if not all(isinstance(word, str) for word in vocabulary.keys()):
        return None
    if not all(isinstance(word_freq, float) for word_freq in vocabulary.values()):
        return None
    if not isinstance(alphabet, list):
        return None
    if not all(isinstance(syb, str) for syb in alphabet):
        return None
    if method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]:
        return None
    if not vocabulary:
        return None
    
    dist_dict = calculate_distance(wrong_word, vocabulary, method, alphabet)
    if not dist_dict:
        return None
    
    dist_min = min(dist_dict.values())

    min_dist_words = [word for word, dist in dist_dict.items() if dist == dist_min]
    if len(min_dist_words) == 1:
        return min_dist_words[0]
    
    



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
        return None


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
        return None
    if not isinstance(alphabet, list):
        return None
    if not all(isinstance(syb, str) for syb in alphabet):
        return None


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
        return None
    if not isinstance(alphabet, list):
        return None
    if not all(isinstance(syb, str) for syb in alphabet):
        return None


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
        return None


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
    if not isinstance(alphabet, list):
        return None
    if not all(isinstance(syb, str) for syb in alphabet):
        return None


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
    if not isinstance(alphabet, list):
        return None
    if not all(isinstance(syb, str) for syb in alphabet):
        return None


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
    if not isinstance(frequencies, dict):
        return None
    if not isinstance(alphabet, list):
        return None
    if not all(isinstance(syb, str) for syb in alphabet):
        return None

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
    if not isinstance(token, str):
        return None
    if not isinstance(candidate, str):
        return None
    if not isinstance(match_distance, int):
        return None
    


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
