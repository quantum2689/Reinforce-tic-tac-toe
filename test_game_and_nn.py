import unittest
import numpy as np
from game import Game
from nn import Network, network


class TestTicTacToe(unittest.TestCase):
    def test_game_initialization(self):
        g = Game()
        self.assertEqual(g.board.shape, (3, 3))
        self.assertFalse(g.game_over())
        self.assertIsNone(g.result())
        self.assertEqual(len(g.legal_moves()), 9)

    def test_invalid_moves_rejected(self):
        g = Game()
        self.assertTrue(g.player_1((0, 0)))
        # Cell already taken
        self.assertFalse(g.player_1((0, 0)))
        self.assertFalse(g.player_2((0, 0)))
        # Out of bounds
        self.assertFalse(g.player_1((3, 0)))
        self.assertFalse(g.player_1((-1, 0)))
        self.assertFalse(g.player_1((0, 3)))
        self.assertFalse(g.player_1((0, -1)))

    def test_row_win(self):
        g = Game()
        g.player_1((0, 0))
        g.player_2((1, 0))
        g.player_1((0, 1))
        g.player_2((1, 1))
        g.player_1((0, 2))
        self.assertTrue(g.game_over())
        self.assertEqual(g.result(), 1)

    def test_draw(self):
        g = Game()
        # X O X
        # X X O
        # O X O
        moves = [
            (1, (0, 0)), (-1, (0, 1)), (1, (0, 2)),
            (-1, (1, 2)), (1, (1, 0)), (-1, (2, 0)),
            (1, (1, 1)), (-1, (2, 2)), (1, (2, 1))
        ]
        for p, pos in moves:
            if p == 1:
                g.player_1(pos)
            else:
                g.player_2(pos)
        self.assertTrue(g.game_over())
        self.assertEqual(g.result(), 0)

    def test_network_forward_and_action_probs(self):
        net = Network()
        board = np.zeros((3, 3))
        board[0, 0] = 1
        probs = net.get_action_probs(board)
        self.assertEqual(probs.shape, (9,))
        # Occupied cell must have 0 probability
        self.assertEqual(probs[0], 0.0)
        # Sum of legal move probs must be approximately 1.0
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_reinforce_step(self):
        net = Network()
        states = np.zeros((4, 9))
        actions = np.array([0, 1, 2, 3])
        advantages = np.array([1.0, -1.0, 1.0, 0.5])
        loss = net.train_reinforce(states, actions, advantages)
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))

    def test_save_load(self):
        net1 = Network()
        net1.save("test_model.npz")
        net2 = Network()
        net2.load("test_model.npz")
        for k in net1.params:
            np.testing.assert_array_almost_equal(net1.params[k], net2.params[k])
        import os
        if os.path.exists("test_model.npz"):
            os.remove("test_model.npz")


if __name__ == "__main__":
    unittest.main()
