"""
Programming 2025.

Seminar 12.

Polymorphism.
"""

# Animal
## Attributes
### eating_type (predator, herbivore)
### habitat
### moving_type
## Methods
### breathe
### eat
### sleep
## Magic Methods
### __str__

# Dog
## Attributes
### eating_type (predator)
### tamed (wild, domesticated)
### breed
### habitat
### moving_type
## Methods
### breathe
### eat
### sleep
## Magic Methods
### __str__

# Cat
## Attributes
### eating_type (predator)
### habitat
### moving_type
### character
### fluffyness
## Methods
### breathe
### eat
### sleep
## Magic Methods
### __str__

# AnimalCollection
## Magic Methods
### __str__
### __len__
### __getitem__
## Advanced
### __setitem__
### __next__
### __iter__

# One code to process different types

# bubble = Cat(habitat='flat', character='playful', fluffiness='very')

# animals = [pug, bubble]
# for animal in animals:
#     animal.eat('meat')


def main() -> None:
    """
    Launch listing.
    """


if __name__ == "__main__":
    main()
