import numpy as np

class Game:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)

    def player_1(self, pos):
        return self._move(pos, 1)

    def player_2(self, pos):
        return self._move(pos, -1)

    def _move(self, pos, player):
        if self.board[pos] == 0:      # only place on an empty cell
            self.board[pos] = player
            return True
        return False

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

g = Game()

g.player_1((0, 0))   # X takes top-left
g.player_2((1, 1))   # O takes center
g.player_1((0, 1))
g.player_2((2, 2))
g.player_1((0, 2))   # X completes the top row

print(g.board)
print(g.win())       # 1 -> player 1 won