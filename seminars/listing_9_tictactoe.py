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

    def __init__(self, size: int = 3) -> None:
        self._size = size
        self._moves_left = size**2
        self._moves: list[Move] = []

    def add_move(self, move: Move) -> bool:
        """
        Add a move to the board if the cell is empty.

        Args:
            move (Move): The move to add

        Returns:
            bool: True if move was added successfully, False otherwise
        """
        for previous_move in self._moves:
            if previous_move.row == move.row and previous_move.col == move.col:
                return False

        self._moves.append(move)
        self._moves_left -= 1
        return True

    def get_moves(self) -> list[Move]:
        """Get all made moves."""
        return self._moves.copy()

    def get_size(self) -> int:
        """Get board size."""
        return self._size

    def get_moves_left(self) -> int:
        """Get number of moves left on the board."""
        return self._moves_left

    def show(self) -> None:
        """Print current state of the board."""
        board = [[" " for _ in range(self._size)] for _ in range(self._size)]

        for move in self._moves:
            board[move.row][move.col] = move.label

        print("Current board:")
        for row in board:
            print("|" + "|".join(row) + "|")
        print()


class Game:
    """
    Store game status and enable moves.

    Instance attributes:
        _size (int): size of playing board (most commonly 3)
        _board (Board): playing board
        _players (tuple[Player, ...]): tuple with 2 players
        _current_player_idx (int): index of the player that should make a move next
        _finished (bool): flag if the game has finished: there was winner or tie

    Instance methods:
        _next_player(self): Update the next player to make a move.
        _check_move(self, move: Move) -> bool: Verify that the move can be made.
        _register_move(self, move: Move) -> bool: Put the move on the playing board.
        _check_for_winner(self) -> bool: Check if win state is present
        play(self, row: int, col: int) -> bool: Process one step of game
    """

    def __init__(self, size: int = 3) -> None:
        self._size = size
        self._board = Board(size)
        self._players = (Player("X"), Player("O"))
        self._current_player_idx = 0
        self._finished = False

    def _next_player(self) -> None:
        """Update the next player to make a move."""
        self._current_player_idx = (self._current_player_idx + 1) % 2

    def _check_move(self, move: Move) -> bool:
        """Verify that the move can be made."""
        if self._finished:
            return False
        if move.row < 0 or move.row >= self._size or move.col < 0 or move.col >= self._size:
            return False
        return True

    def _register_move(self, move: Move) -> bool:
        """Put the move on the playing board."""
        return self._board.add_move(move)

    def _check_for_winner(self) -> bool:
        """Check if win state is present."""
        moves = self._board.get_moves()
        if len(moves) >= self._size * 2 - 1:
            return any(self._check_win_condition(player.label) for player in self._players)
        return False

    def _check_win_condition(self, label: MarkerType) -> bool:
        """
        Check if player with given label has won.
        
        Args:
            label (MarkerType): Player label to check
            
        Returns:
            bool: True if player has won, False otherwise
        """
        moves = [move for move in self._board.get_moves() if move.label == label]

        for row in range(self._size):
            if sum(1 for move in moves if move.row == row) == self._size:
                return True

        for col in range(self._size):
            if sum(1 for move in moves if move.col == col) == self._size:
                return True

        if sum(1 for move in moves if move.row == move.col) == self._size:
            return True
        if sum(1 for move in moves if move.row + move.col == self._size - 1) == self._size:
            return True

        return False

    def play(self, row: int, col: int) -> bool:
        """
        Process one step of game.

        Args:
            row (int): Row index for the move
            col (int): Column index for the move

        Returns:
            bool: True if game continues, False if game finished
        """
        if self._finished:
            return False

        current_player = self._players[self._current_player_idx]
        move = current_player.make_move(row, col)

        if not self._check_move(move):
            return True

        if not self._register_move(move):
            return True

        self._board.show()

        if self._check_for_winner():
            print(f"Player {current_player.label} wins!")
            self._finished = True
            return False

        if self._board.get_moves_left() == 0:
            print("It's a tie!")
            self._finished = True
            return False

        self._next_player()
        return True


def main() -> None:
    """
    Launch tic-tac-toe game.
    """
    print("Created players")

    game = Game(3)
    print("Created game")

    print("Making moves...")

    moves = [(0, 0), (1, 1), (0, 1), (2, 2), (0, 2)]

    for row, col in moves:
        if not game.play(row, col):
            break

    print("Game finished!")


if __name__ == "__main__":
    main()
