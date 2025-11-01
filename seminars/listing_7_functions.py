"""
Programming 2025.

Seminar 7.

Functions.
"""

# Common information about functions
#
# functions are reusable blocks of code designed to perform a specific task
# functions are defined using the `def`` keyword followed by the function name and parentheses
# arguments can be passed to functions within the parentheses
# functions can return data as a result using the return statement


# Create a function, that doesn't return anything
def print_hello_world() -> None:
    """
    Prints a string "Hello world!".

    Returns:
        none: function doesn't return anything
    """
    print("Hello world!")


# Function calls with expected result:
# print_hello_world() -> Hello world!


# Create a function, that returns string
def return_hello_world() -> str:
    """
    Returns a string "Hello world!".

    Returns:
        str: string "Hello world!"
    """
    return "Hello world!"


# Function calls with expected result:
# print(return_hello_world()) -> Hello world!


# By default, functions accept the exact number of arguments
def function_with_two_arguments(arg1: int, arg2: int) -> None:
    """
    Prints 2 received arguments.

    Args:
        arg1 (int): first number
        arg2 (int): second number

    Returns:
        none: function doesn't return anything
    """
    print(f"I received two arguments: {arg1} and {arg2}")


# Function calls with expected result:
# function_with_two_arguments(1)
# -> TypeError: function_with_two_arguments() missing 1 required positional argument: 'arg2'
# function_with_two_arguments(1, 2, 3)
# -> TypeError: function_with_two_arguments() takes 2 positional arguments but 3 were given
# function_with_two_arguments(1, 2) -> I received two arguments: 1 and 2


# Default arguments may or may not be passed
def print_all_arguments(arg1: str, arg2: str, arg3: str = "Argument 3") -> None:
    """
    Prints received arguments.

    Args:
        arg1 (str): first argument
        arg2 (str): second argument
        arg3 (str): third argument

    Returns:
        none: function doesn't return anything
    """
    print(f"I received these arguments: {arg1, arg2, arg3}")


# Function calls with expected result:
# print_all_arguments("Argument 1", "Argument 2")
# -> I received these arguments: Argument 1, Argument 2, Argument 3
# print_all_arguments("Argument 1", "Argument 2", "Argument 4")
# -> I received these arguments: Argument 1, Argument 2, Argument 4


# Positional vs keyword arguments
def who_loves_whom(who: str, whom: str) -> None:
    """
    Prints two received arguments.

    Args:
        who (str): first argument
        whom (str): second argument

    Returns:
        none: function doesn't return anything
    """
    print(f"{who} loves {whom}")


# Function calls with expected result:
# who_loves_whom("mother", "daughter") -> mother loves daughter
# who_loves_whom(whom="mother", who="daughter") -> daughter loves mother


# Built-in functions:
#
# print() -> prints the value into console
# max() -> finds the maximum element in an array
# min() -> finds the minimum element in an array
# dict() -> creates dictionary
# str() -> casts the value to a string
# list() -> creates an empty list or converts value to a list
# type() -> returns type of the value
# etc.


# TASKS


# Task 1:
def calculate_sum() -> int:
    """
    Return sum of received numbers.

    Args:
        arg1 (int): first number
        arg2 (int): second number
        arg3 (int): third number

    Returns:
        int: sum of received numbers
    """
    # student implementation goes here


# Function calls with expected result:
# calculate_sum(1, 2, 3) -> 6
# calculate_sum(1, -5, 0) -> 4


# Task 2:
def calculate_power() -> int:
    """
    Raise the number to the required power.

    Args:
        number (int): number
        power (int): required power

    Returns:
        int: number in the required power
    """
    # student implementation goes here


# Function calls with expected result:
# calculate_power(2, 3) -> 8
# calculate_power(7, 2) -> 49
# calculate_power(1589329, 0) -> 1


# Task 3:
def calculate_factorial() -> int:
    """
    Calculate factorial of the received number.

    Args:
        number (int): number

    Returns:
        int: factorial of the received number
    """
    # student implementation goes here


# Function calls with expected result:
# calculate_factorial(3) -> 6
# calculate_power(2) -> 2
# calculate_power(0) -> 1


# Task 4:
def encode_message() -> list:
    """
    Encode the message.

    Args:
        message (str): string to encode
        encode_dict (dict): dictionary in the form of {character: digit}

    Returns:
        list: list of digits as an encoded message
    """
    # student implementation goes here


# Function calls with expected result:
# encode_message("hello", {"h": 1, "e": 2, "l": 3, "o": 4}) -> [1, 2, 3, 3, 4]
# encode_message("abba", {"a": 1, "b": 2, "c": 3, "d": 4}) -> [1, 2, 2, 1]


# Task 5:
def capitalize_string() -> str:
    """
    Return capitalized version of the string.

    Args:
        input_string (str): string to capitalize

    Returns:
        str: capitalized string
    """
    # student implementation goes here


# Function calls with expected result:
# scream("I love programming on Python") -> I LOVE PROGRAMMING ON PYTHON
# scream("Functions are amazing") -> FUNCTIONS ARE AMAZING


# Task 6
def is_allowed_to_drive() -> bool:
    """
    Check, whether it is allowed to drive.
    By default, 18 is the age when it is allowed to drive a car.

    Args:
        personal_information (dict): dictionary with personal information
        threshold (int): default argument with the age when it is allowed to drive a car

    Returns:
        bool: whether it is allowed to drive
    """
    # student implementation goes here


# Function calls with expected result:
# is_allowed_to_drive({"name": "Kath", "eyes": "blue", "age": 20}, 21) -> False
# is_allowed_to_drive({"name": "Dean", "height": 178, "age": 20}) -> True


# Task 7
def get_fibonacci_sequence() -> list:
    """
    Return Fibonacci sequence of the specified length.

    Args:
        length (int): length of Fibonacci sequence

    Returns:
        list: Fibonacci sequence
    """
    # student implementation goes here


# Function calls with expected result:
# get_fibonacci_sequence(7) -> [1, 1, 2, 3, 5, 8, 13]
# get_fibonacci_sequence(2) -> [1, 1]


# Task 8
def add_numbers() -> int:
    """
    Sum all the provided numbers.

    Args:
        *numbers (int): numbers

    Returns:
        int: sum of the provided numbers
    """
    # student implementation goes here


# Function calls with expected result:
# add_numbers(1, 2, 3) -> 6
# add_numbers(5, 10, 15) -> 30


# Task 9
def print_student_info() -> None:
    """
    Print provided student information.

    Args:
        **info: information about the student

    Returns:
        none: function doesn't return anything
    """
    # student implementation goes here


# Function calls with expected result:
# print_student_info(name="Alice", grade="A", age=20) -> None


# Task 10
def process_data() -> dict:
    """
    Process personal information about student and its performance.

    Args:
        *performance: final marks
        **personal_info: personal information about student

    Returns:
        dict: dictionary in the form of {"performance": ..., "personal_info": ...}
    """
    # student implementation goes here


# Function calls with expected result
# process_data(10, 8, 4, name="Alice", age=25)
# -> {"performance": (10, 8, 4), "personal_info": {"name": "Alice", "age"s: 25}}
