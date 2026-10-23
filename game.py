import numpy as np


class Game:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)

    def player_1(self, pos):
        return self._move(pos, 1)

    def player_2(self, pos):
        return self._move(pos, -1)

    def _move(self, pos, player):
        if self.game_over():          # no moves after the game has ended
            return False
        try:
            r, c = int(pos[0]), int(pos[1])
        except (TypeError, ValueError, IndexError):
            return False
        if not (0 <= r < 3 and 0 <= c < 3):
            return False
        if self.board[r, c] == 0:      # only place on an empty cell
            self.board[r, c] = player
            return True
        return False

    def legal_moves(self):
        """Returns a list of (row, col) tuples of empty cells."""
        return [tuple(pos) for pos in np.argwhere(self.board == 0)]

    def win(self):
        """Returns 1 if player 1 won, -1 if player 2 won, 0 otherwise."""
        lines = np.concatenate([
            self.board.sum(axis=0),                 # 3 column sums
            self.board.sum(axis=1),                 # 3 row sums
            [np.trace(self.board)],                 # main diagonal
            [np.trace(np.fliplr(self.board))],      # anti-diagonal
        ])
        if 3 in lines:
            return 1
        if -3 in lines:
            return -1
        return 0

    def is_full(self):
        """True if there are no empty cells left."""
        return not (self.board == 0).any()

    def game_over(self):
        """True if the game has ended (win or draw)."""
        return self.win() != 0 or self.is_full()

    def result(self):
        """Returns 1 / -1 if a player won, 0 for a draw, None if still in progress."""
        w = self.win()          # check the win first: a win on the last move is not a draw
        if w != 0:
            return w
        if self.is_full():
            return 0
        return None

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)

    def display(self):
        """Returns a string representation of the board."""
        symbols = {1: "X", -1: "O", 0: "."}
        lines = ["  0 1 2"]
        for r in range(3):
            row_str = " ".join(symbols[self.board[r, c]] for c in range(3))
            lines.append(f"{r} {row_str}")
        return "\n".join(lines)

    def __str__(self):
        return self.display()


if __name__ == "__main__":
    g = Game()
    turn = 1
    while not g.game_over():
        print()
        print(g)
        try:
            prompt = f"Player {1 if turn == 1 else 2} ({'X' if turn == 1 else 'O'}) [row col]: "
            coords = input(prompt).split()
            if len(coords) != 2:
                print("Invalid input, please enter row and column (e.g. '1 1').")
                continue
            pos = (int(coords[0]), int(coords[1]))
            ok = g.player_1(pos) if turn == 1 else g.player_2(pos)
        except (ValueError, IndexError):
            print("Invalid input, use two numbers from 0 to 2, e.g. '1 1'.")
            continue
        if ok:                    # only switch turns after a valid move
            turn = -turn
        else:
            print("Invalid move or cell is taken, try again.")

    print()
    print(g)
    r = g.result()
    print("Player 1 (X) wins!" if r == 1 else "Player 2 (O) wins!" if r == -1 else "It's a draw!")