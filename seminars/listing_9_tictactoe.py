"""
Programming 2025.

Seminar 9.

Brainstorm from the lecture on designing a TicTacToe game.
"""

# pylint:disable=too-few-public-methods
from typing import Literal

MarkerType = Literal["X", "O"]


class Move:
    """
    Store information about move: coordinates and label.

    Instance attributes:
        row (int): Index of row
        col (int): Index of column
        label (MarkerType): Label to put

    Instance methods:
        N/A
    """

    def __init__(self, row: int, col: int, label: MarkerType) -> None:
        self.row = row
        self.col = col
        self.label = label


class Player:
    """
    Enable player functionality: store playing label (O or X), make moves.

    Instance attributes:
        label (MarkerType): label to play (O or X)

    Instance methods:
        make_move(self, row: int, col: int) -> Move: Create instance of Move
    """

    def __init__(self, label: MarkerType) -> None:
        self.label = label

    def make_move(self, row: int, col: int) -> Move:
        """Create a new move instance."""
        return Move(row, col, self.label)


class Board:
    """
    Store game status and enable moves.

    Instance attributes:
        _size (int): size of playing board (most commonly 3)
        _moves_left (int): number of empty cells on the playing board
        _moves (list[Move]): already made moves

    Instance methods:
        show(self): Print current state of the board
        add_move(self, move: Move) -> bool: Add new valid move
        get_moves(self) -> list[Move]: Get already made moves
        get_size(self) -> int: Get size of board
        get_moves_left(self) -> int: Get number of moves left
    """


def main() -> None:
    """
    Launch tic-tac-toe game.
    """
    print("Created players")

    print("Making moves...")

    moves = [(0, 0), (1, 1), (0, 1), (2, 2), (0, 2)]

    for move_row, move_col in moves:
        print(f"Playing at row {move_row}, column {move_col}")

    print("Game finished!")


if __name__ == "__main__":
    main()
