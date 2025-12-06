"""
Programming 2025.

Seminar 13.

Exception Handling: try, except, else, finally, EAFP vs LBYL, custom exceptions.
"""

# pylint: disable=duplicate-code, unused-argument, invalid-name, broad-exception-caught, consider-using-with


# Common information about exceptions
#
# Exceptions are runtime error signals.
# When something goes wrong, Python "raises" an exception.
# If the exception is not handled, the program stops.
#
# Basic structure:
#
# try:
#     code that may raise an exception
# except SomeError:
#     code that handles the exception
# else:
#     code executed only if no exception was raised
# finally:
#     code executed ALWAYS (cleanup, closing files etc.)


from typing import Any

print("=== Basic try/except ===")
try:
    number = int("123")
    print("Converted:", number)
except ValueError:
    print("Conversion error occurred!")


# Using exception arguments
print("=== Handling classic exceptions with arguments ===")
try:
    result = 10 / 0
except ZeroDivisionError as error:
    print("ZeroDivisionError:", error.args)  # tuple of arguments


# Multiple except blocks
print("=== Multiple except blocks ===")
try:
    value = int("hello")
except ValueError:
    print("Cannot convert string to integer")
except TypeError:
    print("Invalid type")


# Else block
print("=== Using else block ===")
try:
    x = int("57")
except ValueError:
    print("Conversion failed")
else:
    print("Else block: successful conversion =", x)


# Finally block
print("=== Using finally block ===")
try:
    file = open("example.txt", "w", encoding="utf-8")
    file.write("Hello!")
except OSError:
    print("File operation error")
finally:
    print("Finally block: attempting to close file")
    try:
        file.close()
    except Exception:
        pass


# EAFP vs LBYL
#                             EAFP              LBY
# performance                  + (-)              +
# readability                  -                  +
# race conditions              +                  -
# number of checks (few/many)  +/-                -/+


print("=== EAFP example ===")
try:
    print("Length:", len(10))  # type: ignore[arg-type]
except TypeError:
    print("Object has no length (EAFP)")

print("=== LBYL example ===")
obj = 10
if hasattr(obj, "__len__"):  # check first
    print("Length:", len(obj))
else:
    print("Object has no length (LBYL)")


# Custom exceptions
class StudentDataError(Exception):
    """
    Custom exception for incorrect student data.
    Stores:
        student_id (int)
        field (str)
    """

    def __init__(self, message: str, student_id: int, field: str):
        super().__init__(message)
        self.student_id = student_id
        self.field = field

    def __str__(self) -> str:
        """
        String representation of StudentDataError exception
        """
        return f"{self.args[0]} " f"(student_id={self.student_id}, field='{self.field}')"


def process_student(student: dict) -> None:
    """
    Validate student dictionary.
    Raise StudentDataError if required fields are missing.
    """
    try:
        if "id" not in student:
            raise StudentDataError("Missing ID", student_id=-1, field="id")

        if "name" not in student:
            raise StudentDataError("Missing name", student_id=student["id"], field="name")

        print(f"Student processed: {student['name']}")

    except StudentDataError as e:
        print("Student data error occurred!")
        print("Message:", e.args[0])
        print("Student ID:", e.student_id)
        print("Field:", e.field)
        print("Full text:", e)


print("=== Custom exception example ===")
process_student({"id": 10})
process_student({"name": "Alice"})
process_student({"id": 5, "name": "Bob"})


# TASKS /////////////////////////////////////////////////////////////


# Task 1:
def safe_divide(a: float, b: float) -> float:
    """
    Safely divide a by b.

    Requirements:
        - Catch ZeroDivisionError.
        - Return float('inf') if division by zero occurs.
        - Use else block to return the correct result.
        - ALWAYS print "Operation finished" using finally.
    """
    # student implementation goes here


# safe_divide(10, 2) → 5.0
# safe_divide(5, 0) → inf


# Task 2:
def get_element(lst: list, index: int) -> int:
    """
    Return the element at the given index.

    Requirements:
        - Catch IndexError → return None.
        - Catch TypeError → return None.
        - Print args of the exception.
    """
    # student implementation goes here


# get_element([1, 2, 3], 1) → 2
# get_element([1, 2, 3], 10) → None
# get_element("hello", 0) → None


# Task 3:
class NegativeNumberError(Exception):
    """
    Custom exception raised when a negative number is passed.
    Must store:
        value (int)
        description (str)
    """

    # student implementation goes here


def square_number(n: int) -> int:
    """
    Return n squared.

    Raise NegativeNumberError if n < 0.
    """
    # student implementation goes here


# square_number(5) → 25
# square_number(-3) → raises NegativeNumberError


# Task 4:
def safe_open_file(filename: str) -> str:
    """
    Try to open the file and read its content.

    Requirements:
        - Handle FileNotFoundError.
        - Use else to return file content.
        - ALWAYS close the file in finally.
    """
    # student implementation goes here


# Task 5:
def convert_to_int(item: Any) -> int:
    """
    Convert the item to int.

    Requirements:
        - Catch ValueError and TypeError.
        - Print exception args.
    """
    # student implementation goes here


# convert_to_int("123") → 123
# convert_to_int("abc") → None
# convert_to_int(None) → None


def main() -> None:
    """
    Module entrypoint
    """


if __name__ == "__main__":
    main()
