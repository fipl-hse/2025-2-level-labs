"""
Programming 2025.

Seminar 14.

Scopes: local, enclosing, global, built-in. Decorators.
"""

# pylint: disable=invalid-name, global-statement, redefined-outer-name, unused-variable, unused-argument
from typing import Callable

# LEGB — rule to determine where to look for variable definitions
# L (Local) - variables defined within the function body
# E (Enclosing) - variables in the scope of any enclosing functions, from inner to outer
# G (Global) - variables at the top level of a Python script or module,
# outside of any function definitions
# B (Built-in) - pre-defined (built-in) names, e.g. print(), len(), int, list, etc.

# Decorator - design pattern that allows a user to add new functionality
# to an existing object without modifying its structure.


x = "global"


def outer() -> None:
    """
    Demonstrate local and enclosing scope.
    """
    y = "enclosing"

    def inner() -> None:
        z = "local"
        print("local     :", z)
        print("enclosing :", y)
        print("global    :", x)
        print("built-in  :", len("abc"))

    inner()


outer()

# Change global name
counter = 0


def increment() -> None:
    """
    Demonstrate global name change.
    """
    global counter
    counter += 1


increment()
print("counter =", counter)


# Change enclosing name
def adder(step: int) -> Callable:
    """
    Demonstrate enclosing name change.
    """
    total = 0

    def add() -> int:
        nonlocal total
        total += step
        return total

    return add


inc = adder(3)
print(inc(), inc(), inc())


# Decoratorts
# Simple decorator
def hello_decorator(func: Callable) -> Callable:
    """
    Demonstrate simple decorator.
    """

    def wrapper(*args: tuple, **kwargs: dict):  # type: ignore
        print("Hello!")
        result = func(*args, **kwargs)
        print("Nice to meet you!")
        return result

    return wrapper


@hello_decorator
def greet(name: str) -> None:
    """
    Greet person.
    """
    print(f"My name is {name}")


greet("Misha")


# Parametrized decorator
def repeat(times: int) -> Callable:
    """
    Decorator that repeats the call of the wrapped function.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: tuple, **kwargs: dict) -> None:
            for _ in range(times):
                func(*args, **kwargs)

        return wrapper

    return decorator


@repeat(times=3)
def beep() -> None:
    """
    Beep.
    """
    print("beep")


beep()


# Memoize decorator
def memoize(func: Callable) -> Callable:
    """
    Demonstrate memoize decorator.
    """
    cache = {}

    def wrapper(*args: tuple) -> dict:
        if args in cache:
            return cache[args]  # type: ignore
        result = func(*args)
        cache[args] = result
        return result  # type: ignore

    return wrapper


@memoize
def fib(n: int) -> int:
    """
    Calculate Fibonacci.
    """
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)  # type: ignore


print(fib(35))


# TASKS /////////////////////////////////////////////////////////////


# Task 1:
def use_global() -> int:
    """
    Increase global_var by 5 and return the new value.

    Returns:
        int: The new value of global_var.
    """
    # student implementation goes here


# use_global() -> 5


# Task 2:
def outer_function() -> Callable:  # type: ignore
    """
    Outer function.

     Returns:
        Callable: decorator
    """

    def inner_function() -> int:
        """
        Increase enclosing_var by 10 and return the new value.

        Returns:
            int: The new value of enclosing_var.
        """

    # student implementation goes here


# outer_function()() -> 15


# Task 3:
def use_nonlocal() -> int:
    """
    Define a nested function that modifies the outer function's variable "a"
    by adding 5 using nonlocal keyword and then returns the new value of "a".

    Returns:
        int: The new value of "a".
    """
    # student implementation goes here


# use_nonlocal() -> 15


# Task 4:
def use_global_and_local() -> int:
    """
    Define a function that sets a new local variable "b" to 10, uses the global "b"
    to multiply by 2 and returns the product.

    Returns:
        int: The product of global "b" and 2.
    """
    # student implementation goes here


# use_global_and_local() -> 6


# Task 5:
def logger(func: Callable) -> Callable:
    """
    Log function.

    Requirements:
        - Log the name of the function.
        - Log positional/named arguments.
        - Log the return value.

    Args:
        func (Callable): function to log time

    Returns:
        Callable: decorator
    """
    # student implementation goes here


@logger
def add(x: int, y: int) -> int:
    """
    Add numbers.

    Args:
        int: first number
        int: second number

    Returns:
        int: sum of numbers
    """
    return x + y


# add(5, 3)


# Task 6:
def timer(func: Callable) -> Callable:
    """
    Time function execution.

    Args:
        func (Callable): function to log time

    Returns:
        Callable: decorator
    """
    # student implementation goes here


@timer
def multiply(x: int, y: int) -> int:
    """
    Multiply numbers.

    Args:
        int: first number
        int: second number

    Returns:
        int: multiplication of numbers
    """
    return x * y


# multiply(4, 5)


# Task 7:
def error_handler(default_value: int) -> Callable:
    """
    Handle errors by returning a default value.

    Args:
        default_value (int): specified default value

    Returns:
        Callable: decorator
    """
    # student implementation goes here


# @error_handler(0)
# def divide(x: int, y: int) -> int:
#     """
#     Divide numbers.

#     Args:
#         int: first number
#         int: second number

#     Returns:
#         int: division of numbers
#     """
#     return x / y


# divide(10, 0)


def main() -> None:
    """
    Launch listing.
    """


if __name__ == "__main__":
    main()
