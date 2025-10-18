"""
Lab 2.
"""

# pylint:disable=unused-argument
from typing import Literal

from lab_1_keywords_tfidf.main import (
    check_dict,
    check_list,
)


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

    all_tokens=len(tokens)
    if all_tokens==0:
        return {}

    count_frequency={}
    for token in tokens:
        if token in count_frequency:
            count_frequency[token]+=1
        else:
            count_frequency[token]=1

    dictionary={}
    for element, value in count_frequency.items():
        dictionary[element]=value/all_tokens

    return dictionary

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

    list_out_of_vocab_words=[]
    for words in tokens:
        if words not in vocabulary:
            list_out_of_vocab_words.append(words)

    return list_out_of_vocab_words


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

    if not token and not candidate:
        return 1.0
    if not token or not candidate:
        return 1.0

    tokens_unique=set(token)
    candidate_unique=set(candidate)

    intersection=len(tokens_unique & candidate_unique)
    unification=len(tokens_unique | candidate_unique)

    jaccard_distance=1.0-(intersection / unification)

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
    if (not isinstance(first_token, str) or
    not check_dict(vocabulary, str, float, False) or
    (alphabet is not None and not check_list(alphabet, str, True))):
        return None

    if method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]:
        return None

    if method == "frequency-based" and alphabet is None:
        return {key: 1.0 for key in vocabulary.keys()}

    if method == "frequency-based":
        distance=calculate_frequency_distance(first_token, vocabulary, alphabet or [])
        return distance

    dictionary = {}
    for token in vocabulary:
        if method == "jaccard":
            distance=calculate_jaccard_distance(first_token, token)
            if distance is not None:
                dictionary[token]=distance

        elif method == "levenshtein":
            distance = calculate_levenshtein_distance(first_token, token)
            if distance is not None:
                dictionary[token]=distance

        elif method == "jaro-winkler":
            distance = calculate_jaro_winkler_distance(first_token, token)
            if distance is not None:
                dictionary[token]=distance

    if not dictionary:
        return None

    return dictionary

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
    if not isinstance(wrong_word, str) or not check_dict(vocabulary, str, float, False):
        return None

    if method not in ["jaccard", "frequency-based", "levenshtein", "jaro-winkler"]:
        return None

    if alphabet is not None and not check_list(alphabet, str, True):
        return None

    if not vocabulary:
        return None

    correct_word=calculate_distance(wrong_word, vocabulary, method, alphabet)
    if not correct_word:
        return None

    minimum_result=min(correct_word.values())

    candidates=[]
    for element, value in correct_word.items():
        if value==minimum_result:
            candidates.append(element)

    if len(candidates)==1:
        return candidates[0]

    candidates.sort()
    candidates.sort(key=lambda word: abs(len(word)-len(wrong_word)))

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
    if not isinstance(token_length, int) or not isinstance(candidate_length, int):
        return None

    if token_length < 0 or candidate_length < 0:
        return None

    n=token_length + 1
    m=candidate_length + 1
    matrix=[[0 for _ in range(m)] for _ in range(n)]

    for j in range(m):
        matrix[0][j]= j

    for i in range(n):
        matrix[i][0]= i

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

    token_length=len(token)
    candidate_length=len(candidate)

    matrix=initialize_levenshtein_matrix(token_length, candidate_length)
    if matrix is None:
        return None

    for i in range(1, token_length+1):
        for j in range(1, candidate_length+1):
            if token[i-1] == candidate[j-1]:
                cost=0
            else:
                cost=1

            deleted=matrix[i-1][j]+1
            insertion=matrix[i][j-1]+1
            rechange=matrix[i-1][j-1]+cost

            matrix[i][j]=min(deleted, insertion, rechange)

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

    token_length=len(token)
    candidate_length=len(candidate)

    matrix=fill_levenshtein_matrix(token, candidate)
    if matrix is None:
        return None

    final_levenshtein_distance=matrix[token_length][candidate_length]

    return final_levenshtein_distance


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

    list_of_words=[]
    for i in range(0, len(word)):
        deleted=word[:i]+word[i+1:]
        list_of_words.append(deleted)
    sorted_list=sorted(list_of_words)
    return sorted_list

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

    list_of_words=[]
    for i in range(0, len(word)+1):
        for a in alphabet:
            added=word[:i]+a+word[i:]
            list_of_words.append(added)
    sorted_list=sorted(list_of_words)
    return sorted_list

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

    list_of_words=[]
    for i in range(0, len(word)):
        for a in alphabet:
            replaced=word[:i]+a+word[i+1:]
            list_of_words.append(replaced)
    sorted_list=sorted(list_of_words)
    return sorted_list

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

    list_of_words=[]
    for i in range(0, len(word)-1):
        swaped_adjacent=word[:i]+word[i+1]+word[i]+word[i+2:]
        list_of_words.append(swaped_adjacent)
    sorted_swaped_adjacent=sorted(list_of_words)
    return sorted_swaped_adjacent

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

    if word == "":
        if not alphabet:
            return []
        return sorted(alphabet)

    deleted=delete_letter(word) or []
    added=add_letter(word, alphabet) or []
    replaced=replace_letter(word, alphabet) or []
    swaped_adjacent=swap_adjacent(word) or []

    candidates=deleted+added+replaced+swaped_adjacent

    list_of_candidates: list[str] = []
    for candidate in candidates:
        if candidate not in list_of_candidates:
            list_of_candidates.append(candidate)

    sorted_list_of_candidates=sorted(list_of_candidates)
    return sorted_list_of_candidates

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

    primary_candidates = generate_candidates(word, alphabet)
    if primary_candidates is None:
        return None

    main_candidates = set(primary_candidates)

    for candidate in primary_candidates:
        secondary_candidates = generate_candidates(candidate, alphabet)
        if secondary_candidates is None:
            return None
        main_candidates.update(secondary_candidates)

    return tuple(sorted(main_candidates))

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

    if not isinstance(frequencies, dict) or not check_dict(frequencies, str, float, False):
        return None

    if not check_list(alphabet, str, True):
        return None

    proposed_candidates=propose_candidates(word, alphabet)
    if proposed_candidates is None:
        return {x: 1.0 for x in frequencies.keys()}

    if word == "":
        return {x: 1.0 for x in frequencies.keys()}

    dictionary_of_candidates={}

    for key in frequencies.keys():
        dictionary_of_candidates[key] = 1.0

    if proposed_candidates is not None and word != "":
        for element in proposed_candidates:
            if element in frequencies:
                frequency = frequencies.get(element, 0.0)
                distance = 1.0 - frequency
                dictionary_of_candidates[element]=distance

    return dictionary_of_candidates

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
    if not isinstance(token, str) or not isinstance(candidate, str):
        return None

    if not isinstance(match_distance, int) or match_distance<0:
        return None

    founded_token=[0]*len(token)
    founded_candidate=[0]*len(candidate)
    matches=0

    for i, element in enumerate(token):
        first=max(0, i - match_distance)
        last=min(len(candidate), i+ match_distance+1)

        for j in range(first, last):
            if founded_candidate[j]==0 and candidate[j]==element:
                founded_token[i]=1
                founded_candidate[j]=1
                matches+=1
                break

    token_bool=[bool(el1) for el1 in founded_token]
    candidate_bool=[bool(el2) for el2 in founded_candidate]

    return (matches, token_bool, candidate_bool)

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

    if not check_list(token_matches, bool, False) or not check_list(candidate_matches, bool, False):
        return None

    matches_of_tokens=[]
    matches_of_candidates=[]

    for i, matches1 in enumerate(token_matches):
        if matches1 and i<len(token):
            matches_of_tokens.append(token[i])

    for j, matches2 in enumerate(candidate_matches):
        if matches2 and j<len(candidate):
            matches_of_candidates.append(candidate[j])

    if len(matches_of_tokens)!=len(matches_of_candidates):
        return None

    transposition=0
    for i,token_match in enumerate(matches_of_tokens):
        if matches_of_candidates[i]!=token_match:
            transposition+=1

    final_transposition=transposition//2

    return final_transposition


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

    if matches<0 or transpositions<0:
        return None

    if len(token)==0 or len(candidate)==0:
        return None

    if matches==0:
        return 1.0

    jaro_similarity = (1/3) * (
    matches / len(token)
    + matches / len(candidate)
    + (matches - transpositions) / matches)

    jaro_distance=1-jaro_similarity
    return jaro_distance


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

    if jaro_distance>1 or jaro_distance<0 or prefix_scaling<0:
        return None

    prefix_len=0
    minimum_len=min(len(token), len(candidate))

    for i in range(min(4, minimum_len)):
        if token[i]==candidate[i]:
            prefix_len+=1
        else:
            break

    adjustment=prefix_len*prefix_scaling*(1-(1-jaro_distance))

    return adjustment

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

    if not token or not candidate:
        return 1.0

    distance = max(len(token), len(candidate))//2 - 1
    distance=max(distance, 0)

    result_of_matches = get_matches(token, candidate, distance)
    if result_of_matches is None:
        return None

    matches_count, token_matches, candidate_matches = result_of_matches
    if matches_count == 0:
        return 1.0

    transpositions=count_transpositions(token, candidate, token_matches, candidate_matches)
    if transpositions is None:
        return None

    jaro_distance=calculate_jaro_distance(token, candidate, matches_count, transpositions)
    if jaro_distance is None:
        return None

    adjustment=winkler_adjustment(token, candidate, jaro_distance, prefix_scaling)
    if adjustment is None:
        return None

    jaro_winkler_distance=jaro_distance-adjustment

    return jaro_winkler_distance
