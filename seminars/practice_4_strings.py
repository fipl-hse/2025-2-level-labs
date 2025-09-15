"""
Programming 2025.

Seminar 4.

Data Type: str. Loops
"""

# pylint: disable=invalid-name,unused-argument

# Common information about strings
#
# strings are immutable
# strings are iterable
# strings are case-sensitive

# Create a string
example = "Hello"  # or "Hello"
print(example)

# String concatenation
greeting = example + " there!"
print(greeting)

# String multiplication
several_hellos = example * 5
print(several_hellos)

# String formatting
# .format() method
example = "{} there!".format(example)  # pylint: disable=consider-using-f-string
print(example)
# f-strings
example = f'{greeting} - this is the "greeting" variable.'
print(example)


# String methods (some of them)
# .split() -> split by the delimiter
# .join() - join by the delimiter
# .upper() - uppercase copy of the string
# .lower() - lowercase copy of the string
# .isalpha() - if all the characters in the text are letters
# .strip() - remove the given element (space by default) from the both ends of the string
# .find() - search the string for the specified value (return the index of the first occurrence)

# TASKS


# Task 1:
def multiply_string(input_string: str, how_many: int) -> str:
    """
    Repeat the given string `how_many` times.

    Args:
        input_string (str): String to repeat
        how_many (int): Number of times to repeat

    Returns:
        str: Repeated string
    """
    # student realisation goes here


# multiply_string('Hi', 2) → 'HiHi'
# multiply_string('Hi', 3) → 'HiHiHi'
# multiply_string('Hi', 1) → 'Hi'
# multiply_string('Hi', 0) → ''


# Task 2:
def front_times(input_string: str, how_many: int) -> str:
    """
    Take the first three characters of the string and repeat them `how_many` times.

    Args:
        input_string (str): Input string
        how_many (int): Number of repetitions

    Returns:
        str: Repeated substring
    """
    # student realisation goes here


# front_times('Chocolate', 2) → 'ChoCho'
# front_times('Chocolate', 3) → 'ChoChoCho'
# front_times('Abc', 3) → 'AbcAbcAbc'
# front_times('A', 4) → 'AAAA'
# front_times('', 4) → ''
# front_times('Abc', 0) → ''


# Task 3:
def extra_end(input_string: str) -> str:
    """
    Take the last two characters of the string and repeat them three times.

    Args:
        input_string (str): Input string

    Returns:
        str: Resulting string
    """
    # student realisation goes here


# extra_end('Hello') → 'lololo'
# extra_end('ab') → 'ababab'
# extra_end('Hi') → 'HiHiHi'
# extra_end('Code') → 'dedede'


# Task 4:
def make_abba(first_string: str, second_string: str) -> str:
    """
    Concatenate two strings in the pattern first+second+second+first.

    Args:
        first_string (str): First string
        second_string (str): Second string

    Returns:
        str: Concatenated result
    """
    # student realisation goes here


# make_abba('Hi', 'Bye') → 'HiByeByeHi'
# make_abba('Yo', 'Alice') → 'YoAliceAliceYo'
# make_abba('What', 'Up') → 'WhatUpUpWhat'
# make_abba('', 'y') → 'yy'


# Task 5:
def combo_string(first_string: str, second_string: str) -> str:
    """
    Concatenate strings as shorter+longer+shorter.

    Args:
        first_string (str): First string
        second_string (str): Second string

    Returns:
        str: Concatenated result
    """
    # student realisation goes here


# combo_string('Hello', 'hi') → 'hiHellohi'
# combo_string('hi', 'Hello') → 'hiHellohi'
# combo_string('aaa', 'b') → 'baaab'
# combo_string('', 'bb') → 'bb'
# combo_string('aaa', '1234') → 'aaa1234aaa'
# combo_string('bb', 'a') → 'abba'


# Task 6:
def count_vowels(input_string: str) -> int:
    """
    Count the number of vowels in a string.

    Must use a for-loop over characters.

    Args:
        input_string (str): Input string

    Returns:
        int: Number of vowels
    """
    # student implementation goes here


# count_vowels("hello") → 2
# count_vowels("xyz") → 0
# count_vowels("AEIOU") → 5


# Task 7:
def remove_vowels(input_string: str) -> str:
    """
    Return a copy of the input string without vowels.

    Must use a for-loop and string concatenation.

    Args:
        input_string (str): Input string

    Returns:
        str: String without vowels
    """
    # student implementation goes here


# remove_vowels("hello") → "hll"
# remove_vowels("xyz") → "xyz"


# Task 8:
def count_non_space(input_string: str) -> int:
    """
    Count all characters in the string except spaces.

    Must use for-loop with continue.

    Args:
        input_string (str): Input string

    Returns:
        int: Number of non-space characters
    """
    # student implementation goes here


# count_non_space("a b c") → 3
# count_non_space("   ") → 0


# Task 9:
def find_first_digit(input_string: str) -> str | None:
    """
    Find the first digit in the string and return it.
    If there is no digit, return None.

    Must use for-loop with break.

    Args:
        input_string (str): Input string

    Returns:
        str | None: First digit found, or None if no digits
    """
    # student implementation goes here


# find_first_digit("abc123") → "1"
# find_first_digit("no digits") → None


# Task 10:
def find_repeated_letter(input_string: str) -> str | None:
    """
    Find the first letter that appears twice in a row.

    Must use for-loop with enumerate().

    Args:
        input_string (str): Input string

    Returns:
        str | None: First repeated letter, or None if no repetition
    """
    # student implementation goes here


# find_repeated_letter("hello") → "l"
# find_repeated_letter("world") → None


# Task 11:
def all_words_capitalized(sentence: str) -> bool:
    """
    Return True if every word in the sentence starts with a capital letter.

    Must use for/else loop.

    Args:
        sentence (str): Sentence to check

    Returns:
        bool: True if all words are capitalized, False otherwise
    """
    # student implementation goes here


# all_words_capitalized("Hello World") → True
# all_words_capitalized("Hello world") → False
# all_words_capitalized("") → True


# Task 12:
def is_palindrome(input_string: str) -> bool:
    """
    Check if input_string is a palindrome.

    Must use for-loop with range() to compare characters by index.

    Args:
        input_string (str): Input string

    Returns:
        bool: True if input_string is a palindrome, False otherwise
    """
    # student implementation goes here


# is_palindrome("level") → True
# is_palindrome("hello") → False


# Task 13:
def count_substring_occurrences(text: str, pattern: str) -> int:
    """
    Count how many times `pattern` appears in `text`.

    Must use manual search with for-loop and slicing.

    Args:
        text (str): Text to search in
        pattern (str): Substring to count

    Returns:
        int: Number of times pattern appears in text
    """
    # student implementation goes here


# count_substring_occurrences("banana", "ana") → 1
# count_substring_occurrences("aaaa", "aa") → 3


# Task 14:
def reverse_word(sentence: str) -> str:
    """
    Reverse words in the sentence that are five or more letters long.

    Args:
        sentence (str): Input sentence

    Returns:
        str: Modified sentence
    """
    # student realisation goes here


# reverse_word("Hey fellow warriors") → "Hey wollef sroirraw"
# reverse_word("This is a test") → "This is a test"
# reverse_word("This is another test") → "This is rehtona test"


# Task 15:
def generate_hashtag(input_string: str) -> str:
    """
    Generate a hashtag from a sentence.

    Must start with # and capitalize the first letter of each word.
    Return False if the input is empty or the result is longer than 140 characters.

    Args:
        input_string (str): Input sentence

    Returns:
        str | bool: Hashtag string or False
    """
    # student realisation goes here


# " Hello there thanks for trying my quiz" → "#HelloThereThanksForTryingMyQuiz"
# "    Hello     World   " → "#HelloWorld"
# "" → False


# Task 16: advanced
def string_splosion(input_string: str) -> str:
    """
    Build a string like 'CCoCodCode' from 'Code'.

    Args:
        input_string (str): Input string

    Returns:
        str: Exploded string
    """
    # student realisation goes here


# string_splosion('Code') → 'CCoCodCode'
# string_splosion('abc') → 'aababc'
# string_splosion('ab') → 'aab'
# string_splosion('Kitten') → 'KKiKitKittKitteKitten'
# string_splosion('x') → 'x'


# Task 17: advanced
def string_match(first_string: str, second_string: str) -> int:
    """
    Count matching 2-character substrings at the same positions in two strings.

    Args:
        first_string (str): First string
        second_string (str): Second string

    Returns:
        int: Number of matching substrings
    """
    # student realisation goes here


# string_match('xxcaazz', 'xxbaaz') → 3
# string_match('abc', 'abc') → 2
# string_match('abc', 'axc') → 0
# string_match('he', 'hello') → 1
# string_match('h', 'hello') → 0
