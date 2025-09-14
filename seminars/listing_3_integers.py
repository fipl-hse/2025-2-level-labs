"""
Programming 2025.

Seminar 1.
Integers and float data.
"""

# pylint: disable=invalid-name, unused-argument, redefined-outer-name

# Common information about numbers
#
# integers (int) are whole numbers (positive, negative, or zero)
# floating-point numbers (float) represent decimal values
# numbers are immutable
# numbers are not iterable
# arithmetic operations can be applied: +, -, *, /, //, %, **

# Create numbers
a = 10  # int
b = 3.5  # float
print(a, b)

# Basic arithmetic operations
print(a + b)  # addition
print(a - b)  # subtraction
print(a * b)  # multiplication
print(a / b)  # division (always float)
print(a // 3)  # integer division
print(a % 3)  # modulus (remainder)
print(a**2)  # exponentiation

# Type conversion
print(int(3.9))  # convert float to int → 3
print(float(7))  # convert int to float → 7.0

# Useful functions for numbers (some of them)
# abs(x)       → absolute value of x
# round(x, n)  → round x to n decimal places
# pow(a, b)    → a raised to the power of b
# divmod(a, b) → returns a tuple (a // b, a % b)
# max(a, b, …) → the largest value
# min(a, b, …) → the smallest value
# sum(iterable) → sum of all elements in an iterable

# Common information about loops
#
# Loops allow repeating a block of code multiple times
# Two main types of loops in Python: for and while
# Loops can be nested, and can include break, continue, and else blocks

# for loop over iterable
example_list = [1, 2, 3, 4, 5]
for item in example_list:
    print(item)  # prints each item in the list

# for loop with range
for i in range(5):  # 0, 1, 2, 3, 4
    print(i)

# for loop with start, stop, step
for i in range(1, 10, 2):  # 1, 3, 5, 7, 9
    print(i)

# for loop with enumerate
fruits = ["apple", "banana", "cherry"]
for index, fruit in enumerate(fruits):
    print(index, fruit)  # prints index and value

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

# while loop
count = 0
while count < 5:
    print(count)
    count += 1

# TASKS


# Task 1:
def add_numbers(a: int, b: int) -> int:
    """
    Return the sum of two integers.

    Args:
        a (int): First integer
        b (int): Second integer

    Returns:
        int: Sum of a and b
    """
    # student implementation goes here


# add_numbers(2, 3) → 5
# add_numbers(-5, 10) → 5
# add_numbers(0, 0) → 0


# Task 2:
def average(a: float, b: float, c: float) -> float:
    """
    Calculate the average of three numbers.

    Args:
        a (float): First number
        b (float): Second number
        c (float): Third number

    Returns:
        float: Average value of the three numbers
    """
    # student implementation goes here


# average(1, 2, 3) → 2.0
# average(10, 20, 30) → 20.0
# average(5.5, 6.5, 7.5) → 6.5


# Task 3:
def is_even(n: int) -> bool:
    """
    Check if a number is even.

    Args:
        n (int): Number to check

    Returns:
        bool: True if n is even, False otherwise
    """
    # student implementation goes here


# is_even(2) → True
# is_even(3) → False
# is_even(0) → True
# is_even(-4) → True


# Task 4:
def area_of_circle(radius: float) -> float:
    """
    Calculate the area of a circle.

    Args:
        radius (float): Radius of the circle

    Returns:
        float: Area of the circle
    """
    # student implementation goes here


# area_of_circle(1) → 3.14159...
# area_of_circle(0) → 0
# area_of_circle(2.5) → ~19.63495


# Task 5:
def factorial(n: int) -> int:
    """
    Calculate the factorial of a number.

    Args:
        n (int): Non-negative integer

    Returns:
        int: Factorial of n
    """
    # student implementation goes here


# factorial(0) → 1
# factorial(1) → 1
# factorial(5) → 120


# Task 6:
def power(a: float, b: int) -> float:
    """
    Raise a number to a power.

    Args:
        a (float): Base number
        b (int): Exponent

    Returns:
        float: Result of a raised to the power of b
    """
    # student implementation goes here


# power(2, 3) → 8
# power(5, 0) → 1
# power(2, -2) → 0.25


# Task 7:
def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the Euclidean distance between two points.
    Formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)

    Args:
        x1 (float): x-coordinate of the first point
        y1 (float): y-coordinate of the first point
        x2 (float): x-coordinate of the second point
        y2 (float): y-coordinate of the second point

    Returns:
        float: Euclidean distance between the two points
    """
    # student implementation goes here


# distance(0, 0, 3, 4) → 5.0
# distance(1, 2, 1, 2) → 0.0
# distance(-1, -1, 2, 3) → 5.0


# Task 8 (advanced):
def fibonacci(n: int) -> int:
    """
    Return the n-th Fibonacci number (0-indexed).

    Args:
        n (int): Index in the Fibonacci sequence (0 or greater)

    Returns:
        int: n-th Fibonacci number
    """
    # student implementation goes here


# fibonacci(0) → 0
# fibonacci(1) → 1
# fibonacci(5) → 5
# fibonacci(7) → 13


# Task 9 (advanced):
def is_prime(n: int) -> bool:
    """
    Check if a number is prime.

    Args:
        n (int): Integer greater than or equal to 2

    Returns:
        bool: True if n is prime, False otherwise
    """
    # student implementation goes here


# is_prime(2) → True
# is_prime(15) → False
# is_prime(17) → True
