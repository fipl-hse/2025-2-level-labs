"""
Programming 2025.

Seminar 11.

Inheritance.
"""

# pylint:disable=too-few-public-methods


class Vehicle:
    """
    Represents a general vehicle.

    Instance attributes:
        max_speed (int): The maximum speed of the vehicle.
        colour (str): The colour of the vehicle.

    Instance methods:
        move() -> None:
            Simulates vehicle movement.
    """


class Car:
    """
    Represents a car, which is a type of vehicle.

    Instance attributes:
        max_speed (int): The maximum speed of the car.
        colour (str): The colour of the car.
        fuel (str): The type of fuel used by the car.

    Instance methods:
        move() -> None:
            Simulates car movement.
        stay() -> None:
            Simulates stopping the car.
    """


class Bicycle:
    """
    Represents a bicycle, which is a type of vehicle.

    Instance attributes:
        number_of_wheels (int): The number of wheels of the bicycle.
        colour (str): The colour of the bicycle.
        max_speed (int): The maximum speed of the bicycle.

    Instance methods:
        move() -> None:
            Simulates bicycle movement.
        freestyle() -> None:
            Simulates performing a freestyle trick.
    """


class Aircraft:
    """
    Represents an aircraft, which is a type of vehicle.

    Instance attributes:
        number_of_engines (int): The number of engines of the aircraft.
        colour (str): The colour of the aircraft.
        max_speed (int): The maximum speed of the aircraft.

    Instance methods:
        move() -> None:
            Simulates aircraft movement.
    """


def main() -> None:
    """
    Launch listing.
    """
    print("Created inheritance hierarchy: Vehicle -> Car, Bicycle, Aircraft")


if __name__ == "__main__":
    main()
