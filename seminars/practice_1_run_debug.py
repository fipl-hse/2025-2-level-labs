"""
Programming 2025.

Seminar 1.
Running python application and debugging.
"""

# pylint: disable=invalid-name


# Three main ideas:
# 1. Python is a program that launches another program.
#       We pass the code from main.py into the python virtual machine (PVM),
#           where it is executed. For example, >>> python main.py
# 2. Reading the output in the terminal after executing the program will help you
#       find the error. Everything is told in the terminal output,
#           the main thing is to look carefully.
# 3. Debugging is the first skill of a programmer.
#       It helps you to take a step-by-step look at how the program works.


# Debugging exercise. Debug the program and fix errors:
first_num = 15
second_num = 0

print(f"Numbers: {first_num} and {second_num}")

# Case1: the first exception trigger:
# third_num = first_num / second_num

# Fix the first exception:
second_num = 6
third_num = first_num / second_num
print(f"Numbers: {first_num}, {second_num} and {third_num}")


# Case 2: the second debugging case:
if first_num * second_num == third_num:
    print("First number * second equals third number")
else:
    print("First number * second does not equal third number")

print("Program finished")
