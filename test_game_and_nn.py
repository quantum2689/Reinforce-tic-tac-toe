import os
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


class TestNetwork(unittest.TestCase):
    def test_forward_shape(self):
        net = Network()
        board = np.zeros((3, 3))
        probs = net.forward(board)
        self.assertEqual(probs.shape, (1, 9))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_action_probs_masking(self):
        net = Network()
        board = np.zeros((3, 3))
        board[0, 0] = 1
        probs = net.get_action_probs(board)
        self.assertEqual(probs.shape, (9,))
        # Occupied cell must have 0 probability
        self.assertAlmostEqual(probs[0], 0.0, places=8)
        # Sum of legal move probs must be approximately 1.0
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_three_layer_architecture(self):
        """Verify the network has 3 weight layers (w1, w2, w3)."""
        net = Network()
        self.assertIn("w1", net.params)
        self.assertIn("w2", net.params)
        self.assertIn("w3", net.params)
        self.assertEqual(net.params["w1"].shape, (9, 128))
        self.assertEqual(net.params["w2"].shape, (128, 128))
        self.assertEqual(net.params["w3"].shape, (128, 9))

    def test_reinforce_step_returns_finite(self):
        net = Network()
        states = np.zeros((4, 9))
        actions = np.array([0, 1, 2, 3])
        advantages = np.array([1.0, -1.0, 1.0, 0.5])
        loss = net.train_reinforce(states, actions, advantages)
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isinf(loss))

    def test_entropy_bonus_prevents_collapse(self):
        """With entropy bonus, action probs should not fully collapse to argmax."""
        net = Network(lr=0.1, entropy_coeff=0.1)
        states = np.zeros((20, 9))
        actions = np.full(20, 4)  # always pick center
        advantages = np.ones(20)
        for _ in range(50):
            net.train_reinforce(states, actions, advantages)
        probs = net.get_action_probs(np.zeros(9))
        # With entropy bonus, no legal move should collapse to effectively zero
        legal_probs = probs[probs > 0]
        self.assertTrue(np.min(legal_probs) > 0.001,
                        f"Min prob {np.min(legal_probs):.6f} collapsed to ~0 — entropy bonus failed")

    def test_gradient_clipping(self):
        """Train with huge advantages — should not produce NaN weights."""
        net = Network(lr=0.01, grad_clip=1.0)
        states = np.zeros((4, 9))
        actions = np.array([0, 1, 2, 3])
        advantages = np.array([1e6, -1e6, 1e6, -1e6])
        net.train_reinforce(states, actions, advantages)
        for k, p in net.params.items():
            self.assertFalse(np.any(np.isnan(p)), f"NaN in {k} after clipped update")
            self.assertFalse(np.any(np.isinf(p)), f"Inf in {k} after clipped update")

    def test_lr_decay(self):
        net = Network(lr=1e-3)
        self.assertEqual(net.lr, 1e-3)
        net.set_lr(5e-4)
        self.assertEqual(net.lr, 5e-4)

    def test_save_load(self):
        net1 = Network()
        net1.save("test_model.npz")
        net2 = Network()
        net2.load("test_model.npz")
        for k in net1.params:
            np.testing.assert_array_almost_equal(net1.params[k], net2.params[k])
        if os.path.exists("test_model.npz"):
            os.remove("test_model.npz")

    def test_backward_compat_alias(self):
        """Ensure `network` alias still works."""
        net = network()
        self.assertIsInstance(net, Network)


class TestTraining(unittest.TestCase):
    def test_get_net_move_returns_legal(self):
        from train import get_net_move
        net = Network()
        board = np.zeros((3, 3))
        board[0, 0] = 1
        board[1, 1] = -1
        pos, action = get_net_move(net, board)
        self.assertIsNotNone(pos)
        self.assertEqual(board[pos[0], pos[1]], 0)

    def test_temperature_affects_distribution(self):
        from train import get_net_move
        net = Network()
        board = np.zeros((3, 3))
        # With very low temperature the distribution should be peaked
        probs_cold = net.get_action_probs(board)
        # This is a smoke test — just ensure it doesn't crash
        pos, _ = get_net_move(net, board, temperature=0.1)
        self.assertIsNotNone(pos)
        pos, _ = get_net_move(net, board, temperature=3.0)
        self.assertIsNotNone(pos)

    def test_play_game_runs(self):
        from train import play_game
        net = Network()
        transitions, result = play_game(net)
        self.assertIn(result, [1, -1, 0])
        self.assertGreater(len(transitions), 0)

    def test_evaluate_runs(self):
        from train import evaluate
        net = Network()
        w, d, l = evaluate(net, 10)
        self.assertAlmostEqual(w + d + l, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
