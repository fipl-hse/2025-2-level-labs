"""
Programming 2025.

Seminar 5.

Data Type: str. loops. list. tuple.
"""

# pylint: disable=duplicate-code, unused-argument, invalid-name
import copy

# Common information about strings
#
# strings are immutable
# strings are iterable
# strings are case-sensitive

# Create a string
example_str = "Hello"  # or "Hello"
print(example_str)

greeting = example_str + " there!"
print(greeting)

# String formatting
# .format() method
example_str = "{} there!".format(example_str)  # pylint: disable=consider-using-f-string
print(example_str)
# f-strings
example_str = f'{greeting} - this is the "greeting" variable.'
print(example_str)


# Common information about loops
#
# loops allow repeating a block of code multiple times
# two main types of loops in Python: for and while
# loops can be nested, and can include break, continue, and else blocks

# for loop over iterable
example = [1, 2, 3, 4, 5]
for item in example:
    print(item)  # prints each item in the list

# while loop
count = 0
while count < 5:
    print(count)
    count += 1

# break and continue
for i in range(5):
    if i == 3:
        break  # exit the loop
    if i == 1:
        continue  # skip this iteration
    print(i)

# for / else
for i in range(3):
    print(i)
    if i == 5:
        break
else:
    print("Loop completed without break")


# Common information about lists
#
# lists are mutable
# lists are iterable

# Create a list
example_list = [1, 2, 3]
print(example_list)

# List concatenation, the original list doesn't change
first_list = example_list + [2, 3, 4]
print(example_list)
print(first_list)

# List changes
example_list.append(2)
example_list.extend([2, 3, 4])
print(example_list)

# List copy
first_test = [1, 2, 3, [1, 2, 3]]
test_copy = first_test.copy()
print(first_test, test_copy)
test_copy[3].append(4)
print(first_test, test_copy)

first_test = [1, 2, 3, [1, 2, 3]]
test_deepcopy = copy.deepcopy(first_test)
test_deepcopy[3].append(4)
print(first_test, test_deepcopy)

# List methods
# .insert(index, item) - inserts the given item on the mentioned index
# .remove(item) - removes the first occurrence of the given item from the list
# .pop() or .pop(index) – removes the item from the given index
# (or the last item) and returns that item
# .index(item) – returns the index of the first occurrence
# .sort() – sorts the list in place i.e modifies the original list
# .reverse() – reverses the list in place
# .copy() – returns a shallow copy of the list


# Common information about tuples
#
# tuples are typically defined using parentheses, with elements separated by commas
# tuples are immutable, i.e. their elements cannot be changed after creation
# tuples can contain items of different data types
# tuples support indexing and slicing, but do not support item assignment

# Create a tuple
example_tuple = (1, 2, 3)
print(example_tuple)

# Tuple concatenation: the original tuple doesn't change, and a new tuple is created
first_tuple = example_tuple + (2, 3, 4)
print(example_tuple)
print(first_tuple)

# "Modifying" a tuple requires creating a new one
modified_tuple = example_tuple + (9,)
print(modified_tuple)

# Tuple methods
# .count(item) – returns the number of occurrences of the item
# .index(item) – returns the index of the first occurrence of the item


# TASKS


# Task 1:
def round_number(number: float, decimal_places: int) -> str:
    """
    Display the given number rounded to given decimal places.

    Args:
        number (float): input number
        decimal_places (int): number of decimal places

    Returns:
        str: string with given number rounded to given decimal places
    """
    # student implementation goes here


# Function calls with expected result:
# round_string(3.141592653589793, 3) → '3.142'
# round_string(2.675, 2) → '2.67'
# round_string(-123.4567, 1) → '-123.5'


# Task 2:
def align_string(
    subject: str, price: float, left_align: int, right_align: int, decimal_places: int
) -> str:
    """
    Display the given subject and its rounded price aligned within fixed widths.

    Args:
        subject (str): input subject
        price (float): subject's price
        left_align (int): left alignment width
        right_align (int): right alignment width
        decimal_places (int): number of decimal places

    Returns:
        str: string with given subject and its rounded price aligned within fixed widths
    """
    # student implementation goes here


# Function calls with expected result:
# align_string("apple", 250.998, 10, 5, 2)
# align_string("chicken", 490.51, 5, 5, 1)


# Task 3:
def is_prime(number: int) -> bool:
    """
    Determine whether the given integer is a prime number.

    Args:
        number (int): input number

    Returns:
        bool: whether the number is a prime number
    """
    # student implementation goes here


# Function calls with expected result:
# is_prime(2) → True
# is_prime(15) → False


# Task 4:
def find_first_vowel(input_string: str) -> int:
    """
    Find the index of the first vowel in the string.

    The function should:
      - Use `continue` to skip any character that is not a vowel.
      - If a vowel is found, break the loop and return its index.

    Args:
        input_string (str): input string

    Returns:
        int: index of the first vowel in the string
    """
    # student implementation goes here


# Function calls with expected result:
# find_first_vowel("rhythm") → -1
# find_first_vowel("cryptography") → 5


# Task 5:
def sum_until_zero(numbers: list) -> int:
    """
    Calculate the sum of the positive numbers in the list until a zero is encountered.

    The function should:
      - Use `continue` to skip negative numbers (they are not added to the sum).
      - Use `break` to exit the loop immediately when a zero is encountered.
      - Use the else clause on the for-loop: if no zero is encountered, add 10 to the total sum.

    Args:
        numbers (list): list of numbers

    Returns:
        int: sum of positive numbers in the list
    """
    # student implementation goes here


# Function calls with expected result:
# sum_until_zero([1, 2, -3, 4, 0, 5]) → 7
# sum_until_zero([2, 3, 4]) → 19
# sum_until_zero([0, 5, 7]) → 0
# sum_until_zero([-7, 1, -5, 2, 4]) → 7


# Task 6:
def count_evens(numbers: list) -> int:
    """
    Return the number of even ints in the given array.

    Args:
        numbers (list): list of numbers

    Returns:
        int: the number of even ints
    """
    # student implementation goes here


# Function calls with expected result:
# count_evens([2, 1, 2, 3, 4]) → 3
# count_evens([2, 2, 0]) → 3
# count_evens([1, 3, 5]) → 0


# Task 7:
def sum_numbers(numbers: list) -> int:
    """
    Return the sum of the numbers in the array.

    The function should:
      - Ignore sections of numbers starting with a 6 and
      extending to the next 7 (every 6 will be followed by at least one 7).
      - Return 0 for no numbers.

    Args:
        numbers (list): list of numbers

    Returns:
        int: sum of the numbers in the list
    """
    # student implementation goes here


# Function calls with expected result:
# sum_numbers([1, 2, 2]) → 5
# sum_numbers([1, 2, 2, 6, 99, 99, 7]) → 5
# sum_numbers([1, 1, 6, 7, 2]) → 4


# Task 8:
def create_phone_number(numbers: list) -> str:
    """
    Return a string with a phone number of 10 integers (between 0 and 9) in the input array.

    Args:
        numbers (list): list of numbers

    Returns:
        str: string with a phone number
    """
    # student implementation goes here


# Function calls with expected result:
# create_phone_number([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
# => returns "(123) 456-7890"


# Task 9:
def rotate_list(input_list: list, k: int) -> list:
    """
    Rotate the list by k positions to the right.

    Args:
        input_list (list): list of numbers
        k (int): k positions to rotate the list

    Returns:
        list: rotated list of numbers
    """
    # student implementation goes here


# Function calls with expected result:
# rotate_list([1, 2, 3, 4, 5], 2) → [4, 5, 1, 2, 3]
# rotate_list([1, 2, 3], 4) → [3, 1, 2]


# Task 10:
def flatten_list(nested_list: list) -> list:
    """
    Flatten a nested list into a single list of elements.

    Args:
        nested_list (list): nested list of elements

    Returns:
        list: single list of elements
    """
    # student implementation goes here


# Function calls with expected result:
# flatten_list([1, [2, 3], [[4], 5]]) → [1, 2, 3, 4, 5]
# flatten_list([['a'], [['b'], 'c']]) → ['a', 'b', 'c']


# Task 11:
def find_duplicates(input_list: list) -> list:
    """
    Find and return a list of duplicated elements from the given list.

    Args:
        input_list: input list of elements

    Returns:
        list: list of duplicated elements
    """
    # student implementation goes here


# Function calls with expected result:
# find_duplicates([1, 2, 2, 3, 4, 4, 4, 5]) → [2, 4]
# find_duplicates(['a', 'b', 'a', 'c', 'b']) → ['a', 'b']


# Task 12:
def longest_increasing_subsequence_length(input_list: list) -> int:
    """
    Compute the length of the longest strictly increasing subsequence in the list.

    Args:
        input_list (list): list of numbers

    Returns:
        int: length of the longest strictly increasing subsequence
    """
    # student implementation goes here


# Function calls with expected result:
# longest_increasing_subsequence_length([10, 22, 9, 33, 21, 50, 41, 60]) → 5
# longest_increasing_subsequence_length([3, 10, 2, 1, 20]) → 3


# Task 13:
def remove_elements_at_indices(input_list: list, indices: list) -> list:
    """
    Remove elements from the list at the specified indices.

    Args:
        input_list (list): list of elements
        indices (list): list of indices to remove

    Returns:
        list: list with removed elements
    """
    # student implementation goes here


# Function calls with expected result:
# remove_elements_at_indices([10, 20, 30, 40, 50], [1, 3]) → [10, 30, 50]
# remove_elements_at_indices(['a', 'b', 'c', 'd'], [0, 2]) → ['b', 'd']


# Task 14:
def merge_sorted_lists(input_list_1: list, input_list_2: list) -> list:
    """
    Merge two sorted lists into a single sorted list.

    Args:
        input_list_1 (list): first list of numbers
        input_list_2 (list): second list of numbers

    Returns:
        list: merged sorted list of numbers
    """
    # student implementation goes here


# Function calls with expected result:
# merge_sorted_lists([1, 3, 5], [2, 4, 6]) → [1, 2, 3, 4, 5, 6]
# merge_sorted_lists([0, 10, 20], [5, 15]) → [0, 5, 10, 15, 20]


# Task 15:
def check_exam(correct_answers: list, student_answers: list) -> int:
    """
    Return the grade for the exam according to the given answers.

    The function should:
        - Give +4 for each correct answer.
        - Give -1 for each incorrect answer.
        - Give +0 for each blank answer represented as an empty string.
        - Return 0 if the score < 0.

    Args:
        correct_answers (list): list with correct answers
        student_answers (list): list with student answers

    Returns:
        int: grade for the exam
    """
    # student implementation goes here


# Function calls with expected result:
# check_exam(["a", "a", "b", "b"], ["a", "c", "b", "d"]) → 6
# check_exam(["a", "a", "c", "b"], ["a", "a", "b",  ""]) → 7
# check_exam(["a", "a", "b", "c"], ["a", "a", "b", "c"]) → 16
# check_exam(["b", "c", "b", "a"], ["",  "a", "a", "c"]) → 0


# Task 16:
def show_notification(names: list) -> str:
    """
    Show a notification that certain people liked your post in social media.

    Args:
        names (list): list with names of people

    Returns:
        str: string with a notification
    """
    # student implementation goes here


# Function calls with expected result:
# []                                -->  "no one likes this"
# ["Peter"]                         -->  "Peter likes this"
# ["Jacob", "Alex"]                 -->  "Jacob and Alex like this"
# ["Max", "John", "Mark"]           -->  "Max, John and Mark like this"
# ["Alex", "Jacob", "Mark", "Max"]  -->  "Alex, Jacob and 2 others like this"


# Task 17:
def find_anagrams(word: str) -> list:
    """
    Find all anagrams of the word.

    Two words are anagrams of each other if they both contain the same letters.
        - 'abba' and 'baab' are anagrams
        - 'abba' and 'bbaa' are anagrams
        - 'abba' and 'abbba' are not anagrams
        - 'abba' and 'abca' are not anagrams

    Args:
        word (str): input word

    Returns:
        list: list of anagrams of the word
    """
    # student implementation goes here


# Function calls with expected result:
# find_anagrams('abba') => ['aabb', 'bbaa']
# find_anagrams('racer') => ['carer', 'racer', ...]


# Task 18:
def reverse_tuple(t: tuple) -> tuple:
    """
    Reverse the elements of the tuple.

    Args:
        t (tuple): input tuple

    Returns:
        tuple: tuple with the order of elements reversed
    """
    # student implementation goes here


# Function calls with expected result:
# reverse_tuple((1, 2, 3)) => (3, 2, 1)
# reverse_tuple(('a', 'b', 'c')) => ('c', 'b', 'a')


# Task 19:
def extract_even_numbers(t: tuple) -> tuple:
    """
    Extract all even numbers from a tuple of integers.

    Args:
        t (tuple): input tuple

    Returns:
        tuple: tuple containing only even numbers from the input tuple
    """
    # student implementation goes here


# Function calls with expected result:
# extract_even_numbers((1, 2, 3, 4, 5, 6)) => (2, 4, 6)
# extract_even_numbers((7, 9, 11)) => ()


# Task 20:
def swap_elements(t: tuple, index1: int, index2: int) -> tuple:
    """
    Swap the elements in the tuple at the given indices.

    Args:
        t (tuple): input tuple
        index1 (int): the index of the first element
        index2 (int): the index of the second element

    Returns:
        tuple: tuple with the elements at index1 and index2 swapped
    """
    # student implementation goes here


# Function calls with expected result:
# swap_elements((10, 20, 30, 40), 1, 3) => (10, 40, 30, 20)
# swap_elements(('a', 'b', 'c'), 0, 2) => ('c', 'b', 'a')


# Task 21:
def tuple_intersection(t1: tuple, t2: tuple) -> tuple:
    """
    Find the intersection of two tuples.

    Args:
        t1 (tuple): the first input tuple
        t2 (tuple): the second input tuple

    Returns:
        tuple: tuple containing the common elements between t1 and t2
    """
    # student implementation goes here


# Function calls with expected result:
# tuple_intersection((1, 2, 3, 4), (3, 4, 5, 6)) => (3, 4)
# tuple_intersection(('a', 'b', 'c'), ('b', 'd', 'a')) => ('a', 'b')


# Task 22:
def count_occurrences(t: tuple, element: int | str) -> int:
    """
    Count the number of times an element appears in the tuple.

    Args:
        t (tuple): input tuple
        element: the element whose occurrences need to be counted

    Returns:
        int: the number of times element appears in the tuple
    """
    # student implementation goes here


# Function calls with expected result:
# count_occurrences((1, 2, 3, 2, 2, 4), 2) => 3
# count_occurrences(('apple', 'banana', 'apple'), 'apple') => 2
