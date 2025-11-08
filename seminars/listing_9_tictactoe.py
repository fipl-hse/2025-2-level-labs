"""
Programming 2025.

Seminar 9.

Brainstorm from the lecture on designing a TicTacToe game.
"""

# pylint:disable=too-few-public-methods



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
    def __init__(self, row: int, col: int, label: str):
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
    def __init__(self, label):
        self.label = label

    def make_move(self, row: int, col: int):
        return


class Game:
    """
    Store game status and enable moves.

    Instance attributes:
        _size (MarkerType): size of playing board (most commonly 3)
        _board (MarkerType): playing board (most commonly 3x3)
        _players (tuple[Player, ...]): tuple with 2 players
        _current_player_idx (int): index of the player that should make a move next
        _finished (MarkerType): flag if the game has finished: there was winner or tie

    Instance methods:
        _next_player(self): Update the next player to make a move.
        _check_move(self, ...): Verify that the move can be made.
        _register_move(self, ...): Put the move on the playing board.
        _check_for_winner(self, ...): Check if win state is present
        play(self, ...): Process one step of game
    """
    def __init__(self, size, board, players, current_player_idx: int):
        self._size = size
        self._board = board
        self._players = players
        self._current_player_idx = current_player_idx
        self_finished = False

    def _next_player(self):
        return
    
    def _check_move(self):
        return
    
    def _register_move(self):
        return
    
    def _check_for_winner(self):
        return
    
    def play():
        return


class Board:
    """
    Store game status and enable moves.

    Instance attributes:
        _size (MarkerType): size of playing board (most commonly 3)
        _moves_left (int): number of empty cells on the playing board
        _moves (list[Move]): already made moves


    Instance methods:
        show(self, ...): Print current state of the board
        add_move(self, ...): Add new valid move
        get_moves(self, ...): Get already made moves
        get_size(self, ...): Get size of board

    """
    def __init__(self, size: int = 3):
        self._size = size
        self._moves_left = size ** 2
        self._moves = []

    def show(self):
        print(f"Moves left: {self._moves_left}\nPrevious moves: {self._moves}")

    def add_move(self, move: Move) -> bool:
        for previous_move in self._moves:
            if previous_move.col == move.col and previous_move.row == move.row:
                    print("Error")
                    return False
        else:
            self._moves_left -= 1
            self._moves.append(move)
            return True



def main() -> None:
    """
    Launch tic-tac-toe game.
    """
    # 1. Create players
    print("Created players")

    # 2. Create game
    print("Created game")

    # 3. Make move
    print("Made move")

    # 4. Register move
    print("Registered move")

    # 5. Show current state
    print("Showed current state")


if __name__ == "__main__":
    main()
