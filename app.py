import os
import random
import numpy as np
from flask import Flask, render_template, request, jsonify
from game import Game
from nn import Network, rng
from train import get_net_move, random_player

app = Flask(__name__)

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.npz")


class BaselineTrainer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.net = Network(lr=3e-3)
        self.game = Game()
        self.turn = 1
        self.current_trajectory = []
        self.batch_states = []
        self.batch_actions = []
        self.batch_advantages = []
        self.batch_masks = []
        self.games_played = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.last_loss = 0.0
        self.history_labels = []
        self.history_win_rate = []
        self.history_loss = []
        self.batch_size = 16

    def step(self):
        """Advances by a single move or finishes a game."""
        if self.game.game_over():
            self._finish_game()
            self.game = Game()
            self.turn = 1
            self.current_trajectory = []

        board_before = self.game.board.copy()
        current_player = self.turn
        move_made = None
        probs_dict = {}

        if self.turn == 1:
            state = self.game.board.copy().reshape(-1)
            legal_mask = (state == 0)
            all_probs = self.net.get_action_probs(state, legal_mask)
            for idx, p in enumerate(all_probs):
                if legal_mask[idx]:
                    probs_dict[idx] = round(float(p) * 100, 1)

            pos, action = get_net_move(self.net, self.game.board, deterministic=False)
            if pos is not None and self.game.player_1(pos):
                self.current_trajectory.append((state, action, legal_mask))
                move_made = {"player": "Net (X)", "pos": pos, "action": action}
                self.turn = -1
        else:
            pos = random_player(self.game.board)
            if pos is not None and self.game.player_2(pos):
                move_made = {"player": "Random (O)", "pos": pos}
                self.turn = 1

        game_ended = self.game.game_over()
        game_result = self.game.result() if game_ended else None
        update_happened = False

        if game_ended:
            update_happened = self._finish_game()

        return {
            "board": self.game.board.tolist(),
            "probs": probs_dict,
            "move": move_made,
            "turn": self.turn,
            "game_ended": game_ended,
            "game_result": game_result,
            "games_played": self.games_played,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "win_rate": round(self.wins / max(1, self.games_played) * 100, 1),
            "last_loss": round(self.last_loss, 4),
            "update_happened": update_happened,
        }

    def _finish_game(self):
        r = self.game.result()
        self.games_played += 1
        if r == 1:
            self.wins += 1
            base_reward = 1.0
        elif r == -1:
            self.losses += 1
            base_reward = -1.0
        else:
            self.draws += 1
            base_reward = 0.1

        T = len(self.current_trajectory)
        gamma = 0.95
        for t, (s, a, m) in enumerate(self.current_trajectory):
            g_t = base_reward * (gamma ** (T - 1 - t))
            self.batch_states.append(s)
            self.batch_actions.append(a)
            self.batch_masks.append(m)
            self.batch_advantages.append(g_t)

        update_happened = False
        if len(self.batch_states) >= self.batch_size * 3:
            states = np.array(self.batch_states)
            actions = np.array(self.batch_actions)
            masks = np.array(self.batch_masks)
            advantages = np.array(self.batch_advantages, dtype=float)
            adv_std = np.std(advantages)
            if adv_std > 1e-8:
                advantages = (advantages - np.mean(advantages)) / adv_std
            else:
                advantages = advantages - np.mean(advantages)

            self.last_loss = self.net.train_reinforce(states, actions, advantages, masks)
            self.batch_states = []
            self.batch_actions = []
            self.batch_masks = []
            self.batch_advantages = []
            update_happened = True

            self.history_labels.append(str(self.games_played))
            self.history_win_rate.append(round(self.wins / self.games_played * 100, 1))
            self.history_loss.append(round(float(self.last_loss), 4))
            if len(self.history_labels) > 30:
                self.history_labels.pop(0)
                self.history_win_rate.pop(0)
                self.history_loss.pop(0)

        return update_happened

    def train_fast_batch(self, n_games=20):
        for _ in range(n_games):
            g = Game()
            traj = []
            turn = 1
            while not g.game_over():
                if turn == 1:
                    s = g.board.copy().reshape(-1)
                    m = (s == 0)
                    pos, a = get_net_move(self.net, g.board, deterministic=False)
                    if pos is None or not g.player_1(pos):
                        break
                    traj.append((s, a, m))
                    turn = -1
                else:
                    pos = random_player(g.board)
                    if pos is None or not g.player_2(pos):
                        break
                    turn = 1

            r = g.result()
            self.games_played += 1
            if r == 1:
                self.wins += 1
                base_reward = 1.0
            elif r == -1:
                self.losses += 1
                base_reward = -1.0
            else:
                self.draws += 1
                base_reward = 0.1

            T = len(traj)
            gamma = 0.95
            for t, (s, a, m) in enumerate(traj):
                g_t = base_reward * (gamma ** (T - 1 - t))
                self.batch_states.append(s)
                self.batch_actions.append(a)
                self.batch_masks.append(m)
                self.batch_advantages.append(g_t)

        if self.batch_states:
            states = np.array(self.batch_states)
            actions = np.array(self.batch_actions)
            masks = np.array(self.batch_masks)
            advantages = np.array(self.batch_advantages, dtype=float)
            adv_std = np.std(advantages)
            if adv_std > 1e-8:
                advantages = (advantages - np.mean(advantages)) / adv_std
            else:
                advantages = advantages - np.mean(advantages)

            self.last_loss = self.net.train_reinforce(states, actions, advantages, masks)
            self.batch_states = []
            self.batch_actions = []
            self.batch_masks = []
            self.batch_advantages = []

            self.history_labels.append(str(self.games_played))
            self.history_win_rate.append(round(self.wins / self.games_played * 100, 1))
            self.history_loss.append(round(float(self.last_loss), 4))
            if len(self.history_labels) > 30:
                self.history_labels.pop(0)
                self.history_win_rate.pop(0)
                self.history_loss.pop(0)

        # Update probabilities on current board
        state = self.game.board.copy().reshape(-1)
        legal_mask = (state == 0)
        probs_dict = {}
        if legal_mask.any():
            all_probs = self.net.get_action_probs(state, legal_mask)
            for idx, p in enumerate(all_probs):
                if legal_mask[idx]:
                    probs_dict[idx] = round(float(p) * 100, 1)

        return {
            "board": self.game.board.tolist(),
            "probs": probs_dict,
            "games_played": self.games_played,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "win_rate": round(self.wins / max(1, self.games_played) * 100, 1),
            "last_loss": round(self.last_loss, 4),
            "history_labels": self.history_labels,
            "history_win_rate": self.history_win_rate,
            "history_loss": self.history_loss,
        }


class SelfPlayTrainer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.net = Network(lr=2e-3)
        self.game = Game()
        self.turn = 1
        self.traj_x = []
        self.traj_o = []
        self.batch_states = []
        self.batch_actions = []
        self.batch_advantages = []
        self.batch_masks = []
        self.games_played = 0
        self.x_wins = 0
        self.o_wins = 0
        self.draws = 0
        self.last_loss = 0.0
        self.history_labels = []
        self.history_draw_rate = []
        self.history_x_win_rate = []
        self.history_o_win_rate = []
        self.batch_size = 12

    def step(self):
        if self.game.game_over():
            self._finish_game()
            self.game = Game()
            self.turn = 1
            self.traj_x = []
            self.traj_o = []

        move_made = None
        probs_dict = {}

        # The net plays for the current player
        # From current player's perspective, their pieces are 1, opponent's are -1
        board_perspective = self.game.board if self.turn == 1 else -self.game.board
        state = board_perspective.copy().reshape(-1)
        legal_mask = (state == 0)

        if legal_mask.any():
            all_probs = self.net.get_action_probs(state, legal_mask)
            for idx, p in enumerate(all_probs):
                if legal_mask[idx]:
                    probs_dict[idx] = round(float(p) * 100, 1)

            pos, action = get_net_move(self.net, board_perspective, deterministic=False)
            if pos is not None:
                if self.turn == 1:
                    if self.game.player_1(pos):
                        self.traj_x.append((state, action, legal_mask))
                        move_made = {"player": "Self-Play (X)", "pos": pos, "action": action}
                        self.turn = -1
                else:
                    if self.game.player_2(pos):
                        self.traj_o.append((state, action, legal_mask))
                        move_made = {"player": "Self-Play (O)", "pos": pos, "action": action}
                        self.turn = 1

        game_ended = self.game.game_over()
        game_result = self.game.result() if game_ended else None
        update_happened = False

        if game_ended:
            update_happened = self._finish_game()

        return {
            "board": self.game.board.tolist(),
            "probs": probs_dict,
            "move": move_made,
            "turn": self.turn,
            "game_ended": game_ended,
            "game_result": game_result,
            "games_played": self.games_played,
            "x_wins": self.x_wins,
            "o_wins": self.o_wins,
            "draws": self.draws,
            "x_win_rate": round(self.x_wins / max(1, self.games_played) * 100, 1),
            "o_win_rate": round(self.o_wins / max(1, self.games_played) * 100, 1),
            "draw_rate": round(self.draws / max(1, self.games_played) * 100, 1),
            "last_loss": round(self.last_loss, 4),
            "update_happened": update_happened,
        }

    def _finish_game(self):
        r = self.game.result()
        self.games_played += 1
        if r == 1:
            self.x_wins += 1
            rew_x, rew_o = 1.0, -1.0
        elif r == -1:
            self.o_wins += 1
            rew_x, rew_o = -1.0, 1.0
        else:
            self.draws += 1
            rew_x, rew_o = 0.2, 0.2  # draws in self-play are high quality defense!

        gamma = 0.95
        Tx = len(self.traj_x)
        for t, (s, a, m) in enumerate(self.traj_x):
            self.batch_states.append(s)
            self.batch_actions.append(a)
            self.batch_masks.append(m)
            self.batch_advantages.append(rew_x * (gamma ** (Tx - 1 - t)))

        To = len(self.traj_o)
        for t, (s, a, m) in enumerate(self.traj_o):
            self.batch_states.append(s)
            self.batch_actions.append(a)
            self.batch_masks.append(m)
            self.batch_advantages.append(rew_o * (gamma ** (To - 1 - t)))

        update_happened = False
        if len(self.batch_states) >= self.batch_size * 5:
            states = np.array(self.batch_states)
            actions = np.array(self.batch_actions)
            masks = np.array(self.batch_masks)
            advantages = np.array(self.batch_advantages, dtype=float)
            adv_std = np.std(advantages)
            if adv_std > 1e-8:
                advantages = (advantages - np.mean(advantages)) / adv_std
            else:
                advantages = advantages - np.mean(advantages)

            self.last_loss = self.net.train_reinforce(states, actions, advantages, masks)
            self.batch_states = []
            self.batch_actions = []
            self.batch_masks = []
            self.batch_advantages = []
            update_happened = True

            self.history_labels.append(str(self.games_played))
            self.history_draw_rate.append(round(self.draws / self.games_played * 100, 1))
            self.history_x_win_rate.append(round(self.x_wins / self.games_played * 100, 1))
            self.history_o_win_rate.append(round(self.o_wins / self.games_played * 100, 1))
            if len(self.history_labels) > 30:
                self.history_labels.pop(0)
                self.history_draw_rate.pop(0)
                self.history_x_win_rate.pop(0)
                self.history_o_win_rate.pop(0)

        return update_happened

    def train_fast_batch(self, n_games=20):
        for _ in range(n_games):
            g = Game()
            tx = []
            to = []
            turn = 1
            while not g.game_over():
                board_p = g.board if turn == 1 else -g.board
                s = board_p.copy().reshape(-1)
                m = (s == 0)
                pos, a = get_net_move(self.net, board_p, deterministic=False)
                if pos is None:
                    break
                if turn == 1:
                    if not g.player_1(pos):
                        break
                    tx.append((s, a, m))
                    turn = -1
                else:
                    if not g.player_2(pos):
                        break
                    to.append((s, a, m))
                    turn = 1

            r = g.result()
            self.games_played += 1
            if r == 1:
                self.x_wins += 1
                rew_x, rew_o = 1.0, -1.0
            elif r == -1:
                self.o_wins += 1
                rew_x, rew_o = -1.0, 1.0
            else:
                self.draws += 1
                rew_x, rew_o = 0.2, 0.2

            gamma = 0.95
            Tx = len(tx)
            for t, (s, a, m) in enumerate(tx):
                self.batch_states.append(s)
                self.batch_actions.append(a)
                self.batch_masks.append(m)
                self.batch_advantages.append(rew_x * (gamma ** (Tx - 1 - t)))

            To = len(to)
            for t, (s, a, m) in enumerate(to):
                self.batch_states.append(s)
                self.batch_actions.append(a)
                self.batch_masks.append(m)
                self.batch_advantages.append(rew_o * (gamma ** (To - 1 - t)))

        if self.batch_states:
            states = np.array(self.batch_states)
            actions = np.array(self.batch_actions)
            masks = np.array(self.batch_masks)
            advantages = np.array(self.batch_advantages, dtype=float)
            adv_std = np.std(advantages)
            if adv_std > 1e-8:
                advantages = (advantages - np.mean(advantages)) / adv_std
            else:
                advantages = advantages - np.mean(advantages)

            self.last_loss = self.net.train_reinforce(states, actions, advantages, masks)
            self.batch_states = []
            self.batch_actions = []
            self.batch_masks = []
            self.batch_advantages = []

            self.history_labels.append(str(self.games_played))
            self.history_draw_rate.append(round(self.draws / self.games_played * 100, 1))
            self.history_x_win_rate.append(round(self.x_wins / self.games_played * 100, 1))
            self.history_o_win_rate.append(round(self.o_wins / self.games_played * 100, 1))
            if len(self.history_labels) > 30:
                self.history_labels.pop(0)
                self.history_draw_rate.pop(0)
                self.history_x_win_rate.pop(0)
                self.history_o_win_rate.pop(0)

        # Probabilities on current board
        board_perspective = self.game.board if self.turn == 1 else -self.game.board
        state = board_perspective.copy().reshape(-1)
        legal_mask = (state == 0)
        probs_dict = {}
        if legal_mask.any():
            all_probs = self.net.get_action_probs(state, legal_mask)
            for idx, p in enumerate(all_probs):
                if legal_mask[idx]:
                    probs_dict[idx] = round(float(p) * 100, 1)

        return {
            "board": self.game.board.tolist(),
            "probs": probs_dict,
            "games_played": self.games_played,
            "x_wins": self.x_wins,
            "o_wins": self.o_wins,
            "draws": self.draws,
            "x_win_rate": round(self.x_wins / max(1, self.games_played) * 100, 1),
            "o_win_rate": round(self.o_wins / max(1, self.games_played) * 100, 1),
            "draw_rate": round(self.draws / max(1, self.games_played) * 100, 1),
            "last_loss": round(self.last_loss, 4),
            "history_labels": self.history_labels,
            "history_draw_rate": self.history_draw_rate,
            "history_x_win_rate": self.history_x_win_rate,
            "history_o_win_rate": self.history_o_win_rate,
        }


class SavedModelGame:
    def __init__(self):
        self.net = Network()
        if os.path.exists(MODEL_PATH):
            self.net.load(MODEL_PATH)
        self.user_player = 1  # 1 for X, -1 for O
        self.ai_player = -1
        self.game = Game()
        self.turn = 1
        self.score_user = 0
        self.score_ai = 0
        self.score_draw = 0

    def start_game(self, user_symbol='X'):
        self.game = Game()
        self.turn = 1
        self.user_player = 1 if user_symbol == 'X' else -1
        self.ai_player = -self.user_player

        ai_move = None
        probs = self.get_probs()
        if self.turn == self.ai_player:
            ai_move = self._ai_turn()
            probs = self.get_probs()

        return {
            "board": self.game.board.tolist(),
            "probs": probs,
            "ai_move": ai_move,
            "turn": self.turn,
            "game_ended": self.game.game_over(),
            "game_result": self.game.result(),
            "scores": {"user": self.score_user, "ai": self.score_ai, "draw": self.score_draw}
        }

    def user_move(self, row, col):
        if self.game.game_over():
            return {"error": "Game is already over"}

        if self.turn != self.user_player:
            return {"error": "Not user's turn"}

        pos = (row, col)
        ok = self.game.player_1(pos) if self.user_player == 1 else self.game.player_2(pos)
        if not ok:
            return {"error": "Invalid move or cell is taken"}

        self.turn = self.ai_player
        ai_move = None

        if not self.game.game_over():
            ai_move = self._ai_turn()

        game_ended = self.game.game_over()
        game_result = self.game.result() if game_ended else None

        if game_ended:
            if game_result == self.user_player:
                self.score_user += 1
            elif game_result == self.ai_player:
                self.score_ai += 1
            else:
                self.score_draw += 1

        probs = self.get_probs() if not game_ended else {}

        return {
            "board": self.game.board.tolist(),
            "probs": probs,
            "ai_move": ai_move,
            "turn": self.turn,
            "game_ended": game_ended,
            "game_result": game_result,
            "scores": {"user": self.score_user, "ai": self.score_ai, "draw": self.score_draw}
        }

    def _ai_turn(self):
        board_perspective = self.game.board if self.ai_player == 1 else -self.game.board
        pos, action = get_net_move(self.net, board_perspective, deterministic=True)
        if pos is not None:
            if self.ai_player == 1:
                self.game.player_1(pos)
            else:
                self.game.player_2(pos)
            self.turn = self.user_player
            return {"row": pos[0], "col": pos[1]}
        return None

    def get_probs(self):
        board_perspective = self.game.board if self.ai_player == 1 else -self.game.board
        state = board_perspective.copy().reshape(-1)
        legal_mask = (state == 0)
        probs_dict = {}
        if legal_mask.any():
            all_probs = self.net.get_action_probs(state, legal_mask)
            for idx, p in enumerate(all_probs):
                if legal_mask[idx]:
                    probs_dict[idx] = round(float(p) * 100, 1)
        return probs_dict


class LiveLearningGame:
    def __init__(self):
        self.reset()

    def reset(self, load_checkpoint=False):
        self.net = Network(lr=8e-3)  # Slightly higher learning rate for responsive feedback
        if load_checkpoint and os.path.exists(MODEL_PATH):
            self.net.load(MODEL_PATH)
        self.user_player = 1
        self.ai_player = -1
        self.game = Game()
        self.turn = 1
        self.ai_trajectory = []
        self.games_played = 0
        self.user_wins = 0
        self.ai_wins = 0
        self.draws = 0
        self.last_loss = None
        self.last_update_info = None

    def start_game(self, user_symbol='X'):
        self.game = Game()
        self.turn = 1
        self.ai_trajectory = []
        self.user_player = 1 if user_symbol == 'X' else -1
        self.ai_player = -self.user_player

        ai_move = None
        probs = self.get_probs()
        if self.turn == self.ai_player:
            ai_move = self._ai_turn()
            probs = self.get_probs()

        return {
            "board": self.game.board.tolist(),
            "probs": probs,
            "ai_move": ai_move,
            "turn": self.turn,
            "game_ended": self.game.game_over(),
            "game_result": self.game.result(),
            "stats": {
                "games_played": self.games_played,
                "user_wins": self.user_wins,
                "ai_wins": self.ai_wins,
                "draws": self.draws,
                "last_loss": self.last_loss,
                "last_update_info": self.last_update_info,
            }
        }

    def user_move(self, row, col):
        if self.game.game_over():
            return {"error": "Game is already over"}

        if self.turn != self.user_player:
            return {"error": "Not user's turn"}

        pos = (row, col)
        ok = self.game.player_1(pos) if self.user_player == 1 else self.game.player_2(pos)
        if not ok:
            return {"error": "Invalid move or cell is taken"}

        self.turn = self.ai_player
        ai_move = None

        if not self.game.game_over():
            ai_move = self._ai_turn()

        game_ended = self.game.game_over()
        game_result = self.game.result() if game_ended else None
        learning_update = None

        if game_ended:
            self.games_played += 1
            if game_result == self.user_player:
                self.user_wins += 1
                base_reward = -1.0  # AI lost
            elif game_result == self.ai_player:
                self.ai_wins += 1
                base_reward = 1.0   # AI won
            else:
                self.draws += 1
                base_reward = 0.2   # Draw

            # Learn immediately from this game!
            if self.ai_trajectory:
                learning_update = self._learn_from_game(base_reward)

        probs = self.get_probs() if not game_ended else {}

        return {
            "board": self.game.board.tolist(),
            "probs": probs,
            "ai_move": ai_move,
            "turn": self.turn,
            "game_ended": game_ended,
            "game_result": game_result,
            "learning_update": learning_update,
            "stats": {
                "games_played": self.games_played,
                "user_wins": self.user_wins,
                "ai_wins": self.ai_wins,
                "draws": self.draws,
                "last_loss": self.last_loss,
                "last_update_info": self.last_update_info,
            }
        }

    def _ai_turn(self):
        board_perspective = self.game.board if self.ai_player == 1 else -self.game.board
        state = board_perspective.copy().reshape(-1)
        legal_mask = (state == 0)
        pos, action = get_net_move(self.net, board_perspective, deterministic=False)
        if pos is not None:
            if self.ai_player == 1:
                ok = self.game.player_1(pos)
            else:
                ok = self.game.player_2(pos)

            if ok:
                self.ai_trajectory.append((state, action, legal_mask))
                self.turn = self.user_player
                return {"row": pos[0], "col": pos[1]}
        return None

    def _learn_from_game(self, base_reward):
        T = len(self.ai_trajectory)
        gamma = 0.95
        states = []
        actions = []
        masks = []
        advantages = []
        for t, (s, a, m) in enumerate(self.ai_trajectory):
            states.append(s)
            actions.append(a)
            masks.append(m)
            advantages.append(base_reward * (gamma ** (T - 1 - t)))

        states = np.array(states)
        actions = np.array(actions)
        masks = np.array(masks)
        advantages = np.array(advantages, dtype=float)

        # Perform multiple gradient descent steps to strongly adapt to the user's strategy
        loss = 0.0
        for _ in range(4):
            loss = self.net.train_reinforce(states, actions, advantages, masks)

        self.last_loss = round(float(loss), 4)
        status_msg = "Reinforced winning choices!" if base_reward > 0 else "Penalized mistakes and adapted weights!" if base_reward < 0 else "Learned from defensive tie!"
        self.last_update_info = {
            "reward": base_reward,
            "loss": self.last_loss,
            "moves_learned": T,
            "status": status_msg
        }
        return self.last_update_info

    def get_probs(self):
        board_perspective = self.game.board if self.ai_player == 1 else -self.game.board
        state = board_perspective.copy().reshape(-1)
        legal_mask = (state == 0)
        probs_dict = {}
        if legal_mask.any():
            all_probs = self.net.get_action_probs(state, legal_mask)
            for idx, p in enumerate(all_probs):
                if legal_mask[idx]:
                    probs_dict[idx] = round(float(p) * 100, 1)
        return probs_dict


# Global instances
trainer_baseline = BaselineTrainer()
trainer_selfplay = SelfPlayTrainer()
saved_game = SavedModelGame()
live_learning_game = LiveLearningGame()


@app.route("/")
def index():
    return render_template("index.html")


# Mode 1 API: Baseline Training
@app.route("/api/train/baseline/step", methods=["POST"])
def baseline_step():
    return jsonify(trainer_baseline.step())


@app.route("/api/train/baseline/batch", methods=["POST"])
def baseline_batch():
    data = request.get_json() or {}
    n_games = int(data.get("n_games", 20))
    return jsonify(trainer_baseline.train_fast_batch(n_games))


@app.route("/api/train/baseline/reset", methods=["POST"])
def baseline_reset():
    trainer_baseline.reset()
    return jsonify({"status": "reset", "board": trainer_baseline.game.board.tolist()})


# Mode 2 API: Self-Play Training
@app.route("/api/train/selfplay/step", methods=["POST"])
def selfplay_step():
    return jsonify(trainer_selfplay.step())


@app.route("/api/train/selfplay/batch", methods=["POST"])
def selfplay_batch():
    data = request.get_json() or {}
    n_games = int(data.get("n_games", 20))
    return jsonify(trainer_selfplay.train_fast_batch(n_games))


@app.route("/api/train/selfplay/reset", methods=["POST"])
def selfplay_reset():
    trainer_selfplay.reset()
    return jsonify({"status": "reset", "board": trainer_selfplay.game.board.tolist()})


# Mode 3 API: Play vs Saved Model
@app.route("/api/play/saved/new", methods=["POST"])
def play_saved_new():
    data = request.get_json() or {}
    symbol = data.get("symbol", "X")
    return jsonify(saved_game.start_game(symbol))


@app.route("/api/play/saved/move", methods=["POST"])
def play_saved_move():
    data = request.get_json() or {}
    row = int(data.get("row", 0))
    col = int(data.get("col", 0))
    return jsonify(saved_game.user_move(row, col))


# Mode 4 API: Live Learning vs Human
@app.route("/api/play/live-learning/new", methods=["POST"])
def play_live_new():
    data = request.get_json() or {}
    symbol = data.get("symbol", "X")
    return jsonify(live_learning_game.start_game(symbol))


@app.route("/api/play/live-learning/move", methods=["POST"])
def play_live_move():
    data = request.get_json() or {}
    row = int(data.get("row", 0))
    col = int(data.get("col", 0))
    return jsonify(live_learning_game.user_move(row, col))


@app.route("/api/play/live-learning/reset", methods=["POST"])
def play_live_reset():
    data = request.get_json() or {}
    load_checkpoint = bool(data.get("load_checkpoint", False))
    live_learning_game.reset(load_checkpoint=load_checkpoint)
    return jsonify({"status": "reset", "board": live_learning_game.game.board.tolist()})


# Save Current Model
@app.route("/api/model/save", methods=["POST"])
def save_model():
    data = request.get_json() or {}
    source = data.get("source", "baseline")
    if source == "selfplay":
        trainer_selfplay.net.save(MODEL_PATH)
    elif source == "live":
        live_learning_game.net.save(MODEL_PATH)
    else:
        trainer_baseline.net.save(MODEL_PATH)
    return jsonify({"status": "saved", "path": MODEL_PATH})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
