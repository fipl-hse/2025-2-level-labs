"""
Programming 2025.

Seminar 8.

Introduction to classes.
"""

# pylint: disable=too-few-public-methods
from collections import Counter

counter = Counter("aasalksjdkslaks")
print(counter.most_common())  # [('a', 4), ('s', 4), ('k', 3), ('l', 2), ('j', 1), ('d', 1)]
print(counter.total())  # 15

# what is 'Counter'? what is 'counter'? what are 'most_common' and 'total'?
# =========================================================================

# Class creation:
#  1. Good practice for naming - CamelCase
#  2. __init__ - method for instance creation
#  3. all class methods receive 'self' as the first argument (with few exceptions)


class MyClass:
    """
    Just a regular class.
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        print(f"I created an instance of a class {self}!")


# creating class instance:
my_instance = MyClass()  # I created an instance of a class <__main__.MyClass ...>!


# What happens when we do not specify __init__?
class IsThereInit:
    """
    Mysterious class with no __init__.
    """


instance = IsThereInit()  # instance is created just fine (why?)


# Pretty much everything is an instance of something in Python
print(type(my_instance))  # <class '__main__.MyClass'>
print(isinstance(my_instance, MyClass))  # True

print(type("Well, it is a string"))  # <class 'str'>
print(isinstance("Well, it is a string", str))  # True

print(type(MyClass))  # <class 'type'>
print(type(str))  # <class 'type'>
print(type(type))  # <class 'type'>

# Even classes are instances of another class -- type
#  =============================================================
# Instance attributes and methods


class Animal:
    """
    A class for instantiating different animals with various names, legs count and scariness scores.
    """

    def __init__(self, name: str, legs: int, scariness: int) -> None:
        """
        Initializer called when instance is created, creates instance attributes.
        """
        self.name = name  # attribute
        self.legs = legs  # another attribute
        self.scariness = scariness  # and another attribute

    # instance method
    def introduce(self) -> None:  # do not forget to accept 'self' argument!
        """
        Make animal introduce itself!
        """
        print(f"Hello! My name is {self.name}!")

    # instance method
    # def bad_practice_to_add_attribute_not_in_init(self):
    #     """
    #     Bad practice, let's not do that :(
    #     And pylint won't let it happen either
    #     """
    #     self.bad_practice = 'yeah, we have added an attribute not in __init__... sorry...'


dog = Animal(name="Sharick", legs=4, scariness=8)
print(dog.name, dog.legs, dog.scariness)  # Sharick 4 8
# we do not pass self when calling a method!!!
dog.introduce()  #  Hello! My name is Sharick!

spider = Animal("Spidy", 8, 225)  # Spidy 8 225
print(spider.name, spider.legs, spider.scariness)
spider.introduce()  #  Hello! My name is Spidy!
# spider.bad_practice_to_add_attribute_not_in_init()

print(hasattr(dog, "bad_practice"))  # False
# print(dog.bad_practice)  # Attribute error

print(hasattr(spider, "bad_practice"))  # True
# print(spider.bad_practice)  # yeah, we have added an attribute not in __init__... sorry..


# Arguments can be changed outside the class
dog.legs = 10
dog.scariness = 100000
dog.name = "10-legged Sharick"
print(dog.name, dog.legs, dog.scariness)  # 10-legged Sharick 10 100000
dog.introduce()  # Hello! My name is 10-legged Sharick!


# 'self' is how the instance of a class accesses its fields
class Student:
    """
    A regular student.
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        self.hours_of_sleep = 0
        self.assignments_done = 0
        self.coffee_drunk = 0

    def sleep(self, hours: int) -> None:
        """
        Increases number of hours slept.
        """
        self.hours_of_sleep += hours
        self.coffee_drunk = 0

    def drink_coffee(self, cups: int) -> None:
        """
        Increases number of coffee cups consumed.
        """
        self.coffee_drunk += cups

    def do_homework(self) -> None:
        """
        Increases number of assignments finished.

        Only works if the student is not too tired
        """
        if self.hours_of_sleep < 4:
            if self.coffee_drunk < 2:
                print("Cannot do homework, too tired :(")
                return
            self.coffee_drunk = self.coffee_drunk - 1
        self.hours_of_sleep = self.hours_of_sleep - 2
        self.assignments_done += 1

    def live_a_day(self) -> None:
        """
        Describes a day in life of a student.
        """
        self.sleep(5)
        self.drink_coffee(1)
        self.do_homework()
        self.do_homework()  # Cannot do homework, too tired :(
        self.drink_coffee(1)
        self.do_homework()
        self.do_homework()  # Cannot do homework, too tired :(
        print(f"Today I finished {self.assignments_done} assignments")
        self.sleep(5)


student = Student()
student.live_a_day()

#  ====================================================

print("*" * 15 + " TASKS " + "*" * 15)

# Task 1
# easy level
# class ...:
#
#     def __init__(...):
#         ...

# fill the gaps so that the next line prints: "Oh, no! It is another deadline!"
# deadline = Deadline()


#  Task 2
# easy level
# class ...:
#
#     def ...:
#         ...

# fill the gaps so that the next lines print the corresponding messages
# student1 = Student('Marina', 1)  # Hello! My name is Marina, I'm in year 1
# student2 = Student('Nastya', 2)  # Hello! My name is Nastya, I'm in year 2


# Task 3
# easy level
# class Insect:
#
#     def __init__(...):
#         ...
#
#     def introduce(self):
#         print(f'Hi! My name is {self.name} and I have {self.legs} legs')

# bee = Insect('Bee', 6)
# bee.introduce()  # Hi! My name is Bee and I have 6 legs
#
# spider = Insect('Spider', 8)
# spider.introduce()  # Hi! My name is Spider and I have 8 legs


# Task 4
# medium level
# class Student:
#     """
#     If I have less than three deadlines, my mood is Good!
#     If I have from 3 to 5 deadlines, my mood is So-so...
#     If I have more than 5 deadlines, my mood is Bad!!!
#     """


# implement a class so that the following code works
# student = Student(2)
# print(student.mood())  # Good
# student.deadlines = 4
# print(student.mood())  # So-so
# student.deadlines = 1000
# print(student.mood())  # Bad

# Task 5
# medium level
# class Square:
#
#     def __init__(self, side_length: float) -> None:
#         ...
#
#     def get_area(self) -> float:
#         ...
#
#     def get_perimeter(self) -> float:
#         ...

# square1 = Square(side_length=2)
# print(square1.get_area(), square1.get_perimeter())  # 4 8
#
# square2 = Square(side_length=5.12)
# print(square2.get_area(), square2.get_perimeter())  # 26.2144 20.48


# Task 6
# hard level
# Imagine you have to create a class to represent a learning discipline.
# What information should it contain? The name of the subject? The name of the teacher?
# Number of credits? Number of learning hours? Tasks?
# What methods should be present? What attributes should be added?
# Create such a class and demonstrate its instantiating for at least two different subjects.

# class ???:
#     """
#     A class to represent a learning discipline studied in HSE
#     """
