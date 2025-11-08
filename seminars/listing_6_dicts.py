"""
Programming 2025.

Seminar 6.

Data Type: dict.
"""

# pylint: disable=unused-argument,invalid-name,duplicate-value

# Common information about dictionaries
#
# - Dictionaries are used to store data values in key:value pairs
# - Keys must be immutable (str, int, tuple, etc.)
# - Dictionaries are mutable (can be changed in-place)
# - Since Python 3.7+, dictionaries preserve insertion order

# Create a dict
example = {"brand": "Ford", "model": "Mustang", "year": 1964}
print(example)
print("*" * 30)

# Create a dict (second way)
pair_example = dict([(1, "Hello"), (2, "there")])
print("\nDictionary with each item as a pair: ")
print(pair_example)
print("*" * 30)

# Add a key:value pair
example = {"brand": "Ford", "model": "Mustang", "year": 1964}
example["colour"] = "black"
print(example)
print("*" * 30)

# Remove a key:value pair
example.pop("colour")
print(example)
print("*" * 30)

# Change the value of the given key
example["year"] = 2000
print(example)
print("*" * 30)

# Dict methods (some of them)
# .get(key, default) -> get the value by the given key
# .update(another_dict) -> add key:value pairs from another_dict
# .items() -> returns a list containing a tuple for each key value pair
# .values() -> returns a list of all the values in the dictionary
# .keys() -> returns a list of all the keys in the dictionary

# Sets in Python
#
# - A set is a collection of unique elements (unordered, mutable).
# - Defined using curly braces {} or the set() constructor.
# - Useful for removing duplicates, testing membership, and performing mathematical operations.

# Create a set
example_set = {1, 2, 3, 4, 4, 2}
print("Example set (duplicates removed):", example_set)

# Basic operations
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print("Union:", a | b)  # {1, 2, 3, 4, 5, 6}
print("Intersection:", a & b)  # {3, 4}
print("Difference:", a - b)  # {1, 2}
print("Symmetric diff:", a ^ b)  # {1, 2, 5, 6}

print("*" * 15 + " TASKS " + "*" * 15)


# TASKS


# Task 1:
# easy level
def extract_older_people(people: dict[str, int], threshold: int) -> list[str]:
    """
    Return the names of the people who are older than the given threshold.

    Args:
        people (dict[str, int]): Dictionary mapping names to ages.
        threshold (int): Age threshold.

    Returns:
        list[str]: Names of people with age greater than threshold.
    """
    # student realisation goes here


# Function calls with expected result:
# assert extract_older_people({'Andrej': 22, 'Alexander': 28, 'Irine': 20},
#                             20) == ['Andrej', 'Alexander']
# assert extract_older_people({'Hera': 45, 'Zagreus': 25, 'Zeus': 48}, 30) == ['Hera', 'Zeus']


# Task 2:
# easy level
def sum_values(data: dict[str, int]) -> int | float:
    """
    Find the sum of all items in the dictionary.

    Args:
        data (dict[str, int]): Dictionary with numeric values.

    Returns:
        int | float: Sum of values.
    """
    # student realisation goes here


# Function calls with expected result:
# assert sum_values({'a': 300, 'b': 15, 'c': 430}) == 745


# Task 3
# easy level
def find_key(data: dict[str, int]) -> str:
    """
    Return the key with the maximum value.

    Args:
        data (dict[str, int]): Dictionary mapping keys to numeric values.

    Returns:
        str: Key with the largest value.
    """
    # student realisation goes here


# Function calls with expected result:
# assert find_key({'Andrej': 10000, 'Artyom': 15000, 'Alexander': 100000}) == 'Alexander'


# Task 4
# easy level
def remove_duplicates(data: dict[str, int]) -> dict[str, int]:
    """
    Remove duplicates from the dictionary (keep only the first occurrence of a value).

    Args:
        data (dict[str, int]): Dictionary possibly containing duplicate values.

    Returns:
        dict[str, int]: Dictionary without duplicates.
    """
    # student realisation goes here


# Function calls with expected result
# assert remove_duplicates({
#     'Marat': 10000,
#     'Yaroslav': 15000,
#     'Sasha': 10000}) == {'Yaroslav': 15000}


# Task 5
# medium level
def count_letters(sequence: str) -> dict[str, int]:
    """
    Count how many times each letter appears in the string.
    Case-insensitive (e.g. 'A' and 'a' are treated the same).

    Args:
        sequence (str): Input string.

    Returns:
        dict[str, int]: Dictionary with letters as keys and counts as values.
    """
    # student realisation goes here


# Function calls with expected result:
# assert count_letters('Hello there') == {'h': 2, 'e': 3, 'l': 2, 'o': 1, 't': 1, 'r': 1}


# Task 6
# medium level
def decipher(sentence: str, special_characters: dict[int, str]) -> str:
    """
    Decipher a secret message.

    Rules:
        - Each word starts with an ASCII code that should be converted back
          into its corresponding character.
        - The second and the last letter of each word are swapped.

    Args:
        sentence (str): Encoded sentence.
        special_characters (dict[int, str]): Mapping of ASCII codes to characters.

    Returns:
        str: Deciphered text.
    """
    # student realisation goes here


# Function calls with expected result:
# character_decoded_dict = {72: 'H', 103: 'g', 100: 'd', 82: 'R', 115: 's'}
# assert decipher('72olle 103doo 100ya', character_decoded_dict) == 'Hello good day'
# assert decipher('82yade 115te 103o', character_decoded_dict) == 'Ready set go'


# Task 7
# medium level
def bake_cakes(recipe: dict[str, int], ingredients: dict[str, int]) -> int:
    """
    Calculate how many cakes can be baked given a recipe and available ingredients.

    Args:
        recipe (dict[str, int]): Required ingredients and their amounts.
        ingredients (dict[str, int]): Available ingredients and their amounts.

    Returns:
        int: Maximum number of cakes that can be baked.
    """
    # student realisation goes here


# Function calls with expected result:
# assert bake_cakes({'flour': 500, 'sugar': 200, 'eggs': 1},
#            {'flour': 1200, 'sugar': 1200, 'eggs': 5, 'milk': 200}) == 2
# assert bake_cakes({'apples': 3, 'flour': 300, 'sugar': 150, 'milk': 100, 'oil': 100},
#            {'sugar': 500, 'flour': 2000, 'milk': 2000}) == 0


# Task 8
# easy level
def common_elements(original_list: list[int], secondary_list: list[int]) -> set[int]:
    """
    Find common elements of two lists.

    Args:
        original_list (list[int]): First list of numbers.
        secondary_list (list[int]): Second list of numbers.

    Returns:
        set[int]: Elements that are present in both lists.
    """
    # student realisation goes here


# Function calls with expected result:
# assert common_elements([1, 2, 3, 4], [3, 4, 5]) == {3, 4}
# assert common_elements([10, 20], [30, 40]) == set()


# Task 9
# easy level
def unique_letters(word: str) -> set[str]:
    """
    Find all unique letters in a word (case-insensitive).

    Args:
        word (str): Input word.

    Returns:
        set[str]: Unique lowercase letters.
    """
    # student realisation goes here


# Function calls with expected result:
# assert unique_letters("Banana") == {"b", "a", "n"}
# assert unique_letters("Hello") == {"h", "e", "l", "o"}


# Task 10
# medium level
def are_disjoint(original_list: set[int], secondary_list: set[int]) -> bool:
    """
    Check whether two sets are disjoint (no common elements).

    Args:
        original_list (set[int]): First set.
        secondary_list (set[int]): Second set.

    Returns:
        bool: True if sets have no elements in common, False otherwise.
    """
    # student realisation goes here


# Function calls with expected result:
# assert are_disjoint({1, 2, 3}, {4, 5, 6}) is True
# assert are_disjoint({1, 2, 3}, {3, 4, 5}) is False
