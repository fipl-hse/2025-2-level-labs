"""
Lab 2.
"""

# pylint:disable=unused-argument
from typing import Literal
from math import floor, ceil


def check_list(user_input, elements_type: type, can_be_empty: bool) -> bool:
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
    if can_be_empty:
        if user_input == []:
            return True
    else:
        if user_input == []:
            return False
    return all(isinstance(step1, elements_type) for step1 in user_input)

def check_dict(user_input, key_type: type, value_type: type, can_be_empty: bool) -> bool:
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
    if user_input == {}:
        return can_be_empty
    if all(isinstance(step1, key_type) for step1 in user_input.keys()):
        return all(isinstance(step2, value_type) for step2 in user_input.values())
    return False


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
    dict_frequency = {}
    if isinstance(tokens, list) and all(isinstance(step1, str) for step1 in tokens):
        for i in tokens:
            dict_frequency.update({i : (tokens.count(i))/len(tokens)})
        if dict_frequency == {}:
            return None
        return dict_frequency
    return None


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
    if not (check_list(tokens, str, False) and check_dict(vocabulary, str, float, False)):
        return None
    incorrect_words_lst = []
    for i in tokens:
        if i in vocabulary:
            continue
        else:
            incorrect_words_lst.append(i)
    return incorrect_words_lst


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
    if not (isinstance(token, str) and isinstance(candidate, str)):
        return None
    if token == "" and candidate == "":
        return 1.0
    sym = set(token).symmetric_difference(set(candidate))
    union = set(token).union(set(candidate))
    return len(sym)/ len(union)


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
    if not (isinstance(token_length, int) and isinstance(candidate_length, int) and token_length >= 0 and candidate_length >= 0):
        return None
    matrix2d = [[0 for i in range(candidate_length+1)] for j in range(token_length+1)]
    if token_length == 0:
        for i in range(candidate_length+1):
            matrix2d[0][i] = i
    else:
        for i in range(token_length+1):
            matrix2d[i][0] = i
        for j in range(candidate_length+1):
            matrix2d[0][j] = j
    return matrix2d


def fill_levenshtein_matrix(token: str, candidate: str) -> list[list[int]] | None:
    """
    Fill a Levenshtein matrix with edit distances between all prefixes.

    Args:
        token (str): First string.
        candidate (str): Second string.

    Returns:
        list[list[int]] | None: Completed Levenshtein distance matrix.
    """
    if not (isinstance(token, str) and isinstance(candidate, str)):
        return None
    matrix2d = initialize_levenshtein_matrix(len(token), len(candidate))
    for i in range(1, len(token) + 1):
        for j in range(1, len(candidate) + 1):
            if token[i - 1] == candidate[j - 1]:
                matrix2d[i][j] = matrix2d[i - 1][j - 1]
            else:
                matrix2d[i][j] = 1 + min(matrix2d[i][j - 1], matrix2d[i - 1][j], matrix2d[i - 1][j - 1])
    return matrix2d


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
    if not (isinstance(token, str) and isinstance(candidate, str)):
        return None
    matrix2d = fill_levenshtein_matrix(token, candidate)
    return matrix2d[len(token)][len(candidate)]


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
    new_word = ''
    new_words = []
    for i in range(len(word)):
        new_word = word[0:i] + word[i+1:]
        new_words.append(new_word)
    sorted_words = sorted(new_words)
    return sorted_words

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
    if not (isinstance(word, str) and isinstance(alphabet, list) and check_list(alphabet, str, False)):
        return []
    new_words = []
    word_lst = list(word)
    for i in range (len(word)):
        for j in alphabet:
            word_lst.insert(i, j)
            new_word = ''.join(word_lst)
            new_words.append(new_word)
            new_word = ''
            word_lst = list(word)
    #what's wrong with this?
    return new_words

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
    if not (isinstance(word, str) and isinstance(alphabet, list) and check_list(alphabet, str, False)):
        return []
    new_word = ''
    new_words = []
    for i in range(len(word)):
        for j in alphabet:
            new_word = word.replace(word[i], j)
            new_words.append(new_word)
    sorted_words = sorted(new_words)
    return sorted_words

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
    new_word = ''
    new_words = []
    # # for i in range(0, len(word), 2):
    # #     new_word += word[i+1] + word[i]
    # #     new_words.append(new_word)
    # for i in range(0, len(word), 2):
    #     new_word = ''.join([ word[i:i+2][::-1]])
    #     new_words.append(new_word)
    # return new_words
    for i in range(len(word)):
        one_letter = word[i]
        if i <= len(word):
            two_letter = word[i]
        else:
            two_letter = ''
        if i == 0:
            new_word = two_letter + one_letter + word[2:]
        else:
            new_word = word[0:i-1] + two_letter + one_letter + word[i:]
            new_words.append(new_word)
    return new_words
print(swap_adjacent("word"))

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
    if not (isinstance(token, str) and isinstance(candidate, str) and isinstance(match_distance, int)
            and match_distance >= 0):
        return None
    matches = 0
    lst_tru1 = [False] * len(token)
    lst_tru2 = [False] * len(candidate)
    for i in range(len(token)):
        for j in range(max(0, i - match_distance), 
                       min(len(candidate), i + match_distance + 1)):
            if (token[i] == candidate[j] and lst_tru2[j] == False):
                lst_tru1[i] = True
                lst_tru2[j] = True
                matches += 1
                break
    return matches, lst_tru1, lst_tru2

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
    if not (isinstance(token, str) and isinstance(candidate, str) 
            and isinstance(token_matches, list) and isinstance(candidate_matches, list) 
            and check_list(token_matches, bool, False) and check_list(candidate_matches, bool, False)):
        return None
    transpositions = 0
    point = 0
    for i in range(len(token)):
        if (token_matches[i]):
            while (candidate_matches[point] == False):
                point += 1

            if (token[i] != candidate[point]):
                transpositions += 1
            point += 1
    transpositions = transpositions//2
    return transpositions

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
    if not (isinstance(token, str) and isinstance(candidate, str) 
            and isinstance(matches, int) and isinstance(transpositions, int)
            and matches >= 0 and transpositions >= 0):
        return None
    if matches == 0:
        return 1.0 #for some reasons tests ask to return 1 here when no matching
    if token == candidate:
        return 0.0 #for some reasons tests ask to return 0 here when matching
    jaro_distance = ((matches/len(token) + (matches/len(candidate)) + (matches - transpositions/2)/matches) / 3.0)
    #why does this not work? the formula is correct and when 
    #I use it with any known examples, it works just fine. 
    # It's only the tests that are complaining
    # and at this point I don't know what else I can do
    return round(jaro_distance, 4)

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
    if not (isinstance(token, str) and isinstance(candidate, str) and isinstance(jaro_distance, float) 
            and isinstance(prefix_scaling, float)):
        return None
    match_prefix = 0
    max_distance = 4
    if len(token) <= max_distance or len(candidate) <= max_distance:
        max_distance = min(len(token), len(candidate))
    for i in range(max_distance):
        if token[i] == candidate[i]:
            match_prefix += 1
    adjust = match_prefix * prefix_scaling * prefix_scaling * (1 - jaro_distance)
    #any advice? answer almost correct
    # I don't know what to do
    return adjust

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
    if not (isinstance(token, str) and isinstance(candidate, str) and isinstance(prefix_scaling, float)):
        return None
    matches = get_matches(token, candidate)[0]
    transpositions = count_transpositions(token, candidate)
    jaro_distance = calculate_jaro_distance(token, candidate, matches, transpositions)
    winkler_adjust = winkler_adjustment(token, candidate, jaro_distance, prefix_scaling)
    jaro_winkler = 1 - winkler_adjust
    return jaro_winkler
