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
    def __init__(self, colour: str, max_speed: int):
        self.colour = colour
        self.max_speed = max_speed

    def move(self) -> None:
        print(f"vehicles is moving")
       



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
    def __init__(self, max_speed: int, colour: str, fuel: str):
        Vehicle.__init__(self, max_speed, colour)
        self.fuel = fuel

    def move(self) -> None:
        print(f"{self.fuel} car is moving")
    
    def stay(self) -> None:
        print(f"car is staying")



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
    def __init__(self, number_of_wheels: int, colour: str, max_speed: int):
        Vehicle.__init__(self, max_speed, colour)
        self.number_of_wheels = number_of_wheels

    def move(self) -> None:
        print(f"{self.number_of_wheels} bicycle is moving")
              
    def freestyle(self) -> None:
        print(f"biciycle is making freestyle")

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
    def __init__(self, number_of_engines: int, colour: str, max_speed: int):
        Vehicle.__init__(self, max_speed, colour)
        self.number_of_engines = number_of_engines

    def move(self) -> None:
        print(f"{self.number_of_engines} aircraft is moving")


def main() -> None:
    """
    Launch listing.
    """
    car = Car(180, "red", 100)
    bicycle = Bicycle(30, "blue", 2)
    aircraft = Aircraft(900, "white", 4)
    vehicles = [car, bicycle, aircraft]
    
    for i in vehicles:
        i.move()

    car.stay()
    bicycle.freestyle()
    print("Created inheritance hierarchy: Vehicle -> Car, Bicycle, Aircraft")

if __name__ == "__main__":
    main()
