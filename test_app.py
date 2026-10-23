import unittest
from app import app


class TestFlaskEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"REINFORCE Tic-Tac-Toe", response.data)

    def test_baseline_endpoints(self):
        # Step
        res = self.client.post("/api/train/baseline/step")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("board", data)
        self.assertIn("probs", data)

        # Batch
        res_batch = self.client.post("/api/train/baseline/batch", json={"n_games": 2})
        self.assertEqual(res_batch.status_code, 200)

        # Reset
        res_reset = self.client.post("/api/train/baseline/reset")
        self.assertEqual(res_reset.status_code, 200)

    def test_selfplay_endpoints(self):
        # Step
        res = self.client.post("/api/train/selfplay/step")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("board", data)
        self.assertIn("x_win_rate", data)

        # Batch
        res_batch = self.client.post("/api/train/selfplay/batch", json={"n_games": 2})
        self.assertEqual(res_batch.status_code, 200)

    def test_play_saved_endpoints(self):
        # Start game as X
        res = self.client.post("/api/play/saved/new", json={"symbol": "X"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("board", data)

        # User move
        res_move = self.client.post("/api/play/saved/move", json={"row": 1, "col": 1})
        self.assertEqual(res_move.status_code, 200)
        data_move = res_move.get_json()
        self.assertIn("board", data_move)

    def test_live_learning_endpoints(self):
        # Start game
        res = self.client.post("/api/play/live-learning/new", json={"symbol": "X"})
        self.assertEqual(res.status_code, 200)

        # Make user move
        res_move = self.client.post("/api/play/live-learning/move", json={"row": 0, "col": 0})
        self.assertEqual(res_move.status_code, 200)

        # Reset
        res_reset = self.client.post("/api/play/live-learning/reset", json={"load_checkpoint": False})
        self.assertEqual(res_reset.status_code, 200)


if __name__ == "__main__":
    unittest.main()
