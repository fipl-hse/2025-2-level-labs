"""
Programming 2025.

Seminar 3.
Conditional statements (if, elif, else).
"""

# pylint: disable=invalid-name, unused-argument, redefined-outer-name, chained-comparison

# Common information about conditional statements
#
# if condition:
#     block of code
# elif another_condition:
#     block of code
# else:
#     block of code
#
# Conditions are expressions that evaluate to True or False (boolean values).
# Indentation is important: code inside if/elif/else must be indented.
#
x = 10
if x > 0:
    print("Positive")
elif x == 0:
    print("Zero")
else:
    print("Negative")

# Comparison operators
# ==   → equal
# !=   → not equal
# >    → greater than
# <    → less than
# >=   → greater than or equal
# <=   → less than or equal

# Logical operators
# and  → True if both conditions are True
# or   → True if at least one condition is True
# not  → negation (inverts True/False)

# Order of evaluation (precedence):
# 1. not
# 2. and
# 3. or
#
# Example:
print(True or False and False)  # → True, because "and" is evaluated first
print((True or False) and False)  # → False, parentheses change the order

# The walrus operator (:=)
#
# Introduced in Python 3.8
# It allows assignment inside an expression.
#
# Syntax: variable := expression
#
# Example:
if (n := len("hello")) > 3:
    print(f"Length is {n}")  # → prints "Length is 5"
#
# Without walrus:
n = len("hello")
if n > 3:
    print(f"Length is {n}")

# Long conditions
#
# Sometimes conditions can become too long to fit in one line.
# There are two main ways to split them:

# 1. Using backslash (\) for explicit line continuation
age = 25
country = "USA"
has_permission = True

if age > 18 and country == "USA" and has_permission and age < 30:
    print("Access granted (with backslash)")

# 2. Using parentheses () for implicit line continuation
#    This is more Pythonic and recommended.
if age > 18 and country == "USA" and has_permission and age < 30:
    print("Access granted (with parentheses)")

# Note:
# - Parentheses automatically allow line breaks without backslashes.
# - Recommended style: use parentheses instead of backslashes.

# TASKS


# Task 1:
def is_positive(n: int) -> bool:
    """
    Check if a number is positive.

    Args:
        n (int): Number to check

    Returns:
        bool: True if n > 0, False otherwise
    """
    # student implementation goes here


# is_positive(5) → True
# is_positive(-3) → False
# is_positive(0) → False


# Task 2:
def number_sign(n: int) -> str:
    """
    Return whether a number is positive, negative, or zero.

    Args:
        n (int): Number to classify

    Returns:
        str: "positive", "negative", or "zero"
    """
    # student implementation goes here


# number_sign(5) → "positive"
# number_sign(-3) → "negative"
# number_sign(0) → "zero"


# Task 3:
def max_of_two(a: int, b: int) -> int:
    """
    Return the maximum of two numbers.

    Args:
        a (int): First number
        b (int): Second number

    Returns:
        int: The larger of a and b
    """
    # student implementation goes here


# max_of_two(3, 7) → 7
# max_of_two(10, 2) → 10
# max_of_two(5, 5) → 5


# Task 4:
def grade(score: int) -> str:
    """
    Convert a numeric score to a grade:
    - 90–100 → "A"
    - 80–89 → "B"
    - 70–79 → "C"
    - 60–69 → "D"
    - below 60 → "F"

    Args:
        score (int): Numeric score between 0 and 100

    Returns:
        str: Letter grade
    """
    # student implementation goes here


# grade(95) → "A"
# grade(72) → "C"
# grade(59) → "F"


# Task 5:
def is_in_range(n: int, low: int, high: int) -> bool:
    """
    Check if n is within the inclusive range [low, high].

    Args:
        n (int): Number to check
        low (int): Lower bound
        high (int): Upper bound

    Returns:
        bool: True if low <= n <= high, False otherwise
    """
    # student implementation goes here


# is_in_range(5, 1, 10) → True
# is_in_range(0, 1, 10) → False
# is_in_range(10, 1, 10) → True


# Task 6:
def complex_condition(a: bool, b: bool, c: bool) -> bool:
    """
    Evaluate a logical expression with given boolean values.
    Expression: (a and b) or (not c)

    Args:
        a (bool): First condition
        b (bool): Second condition
        c (bool): Third condition

    Returns:
        bool: Result of the expression
    """
    # student implementation goes here


# complex_condition(True, True, False) → True
# complex_condition(False, True, True) → False
# complex_condition(False, False, False) → True


# Task 7 (advanced):
def leap_year(year: int) -> bool:
    """
    Check if a year is a leap year.
    Rule:
    - divisible by 4 → leap year
    - divisible by 100 → not a leap year
    - divisible by 400 → leap year

    Args:
        year (int): Year to check

    Returns:
        bool: True if leap year, False otherwise
    """
    # student implementation goes here


# leap_year(2000) → True
# leap_year(1900) → False
# leap_year(2024) → True
