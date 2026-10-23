import random
import numpy as np
from game import Game
from nn import Network, rng


def get_net_move(net, board, deterministic=False):
    """Selects a move for the neural network on the given board."""
    state = np.asarray(board, dtype=float).reshape(-1)
    legal_mask = (state == 0)
    if not legal_mask.any():
        return None, None

    probs = net.get_action_probs(state, legal_mask)
    if deterministic:
        action = int(np.argmax(probs))
    else:
        action = int(rng.choice(9, p=probs))

    pos = (action // 3, action % 3)
    return pos, action


def random_player(board):
    """Selects a random legal move from empty cells."""
    empty = [tuple(pos) for pos in np.argwhere(board == 0)]
    return random.choice(empty) if empty else None


def play_game(net, explore=True):
    """
    Plays one game: Net (Player 1 / X) vs Random (Player 2 / O).
    Returns:
        transitions: list of (state, action, legal_mask) for each move the net made
        result: 1 (net won), -1 (net lost), 0 (draw)
    """
    g = Game()
    transitions = []
    turn = 1

    while not g.game_over():
        if turn == 1:
            state = g.board.copy().reshape(-1)
            legal_mask = (state == 0)
            pos, action = get_net_move(net, g.board, deterministic=not explore)
            if pos is None:
                break
            if g.player_1(pos):
                transitions.append((state, action, legal_mask))
                turn = 2
            else:
                break
        else:
            pos = random_player(g.board)
            if pos is None:
                break
            if g.player_2(pos):
                turn = 1
            else:
                break

    return transitions, g.result()


def evaluate(net, n_games=100):
    """Evaluates the network deterministically against the random player."""
    wins = 0
    draws = 0
    losses = 0
    for _ in range(n_games):
        _, result = play_game(net, explore=False)
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1
    return wins / n_games, draws / n_games, losses / n_games


def train(net=None, n_games_per_batch=64, target_win_rate=0.90, gamma=0.95, max_batches=500):
    """
    Trains the network using REINFORCE (policy gradient) against a random player.
    Saves the trained model to 'model.npz' upon reaching the target win rate.
    """
    if net is None:
        net = Network(lr=3e-3)

    print("Starting REINFORCE training (Net as Player 1 vs Random Player)...")
    print(f"Batch size: {n_games_per_batch} games | Discount factor gamma: {gamma}")

    recent_win_rates = []

    for batch_idx in range(1, max_batches + 1):
        batch_states = []
        batch_actions = []
        batch_advantages = []
        batch_masks = []

        batch_wins = 0
        batch_draws = 0
        batch_losses = 0

        for _ in range(n_games_per_batch):
            transitions, result = play_game(net, explore=True)
            if result == 1:
                batch_wins += 1
                base_reward = 1.0
            elif result == -1:
                batch_losses += 1
                base_reward = -1.0
            else:
                batch_draws += 1
                base_reward = 0.1  # small positive reward for a draw instead of a loss

            # Discounted returns for each move in this game trajectory
            T = len(transitions)
            for t, (s, a, m) in enumerate(transitions):
                # Earlier moves discounted, final move gets full base reward
                g_t = base_reward * (gamma ** (T - 1 - t))
                batch_states.append(s)
                batch_actions.append(a)
                batch_masks.append(m)
                batch_advantages.append(g_t)

        if not batch_states:
            continue

        batch_states = np.array(batch_states)
        batch_actions = np.array(batch_actions)
        batch_masks = np.array(batch_masks)
        advantages = np.array(batch_advantages, dtype=float)

        # Baseline subtraction / standardization for variance reduction
        adv_std = np.std(advantages)
        if adv_std > 1e-8:
            advantages = (advantages - np.mean(advantages)) / adv_std
        else:
            advantages = advantages - np.mean(advantages)

        # Update policy network
        loss = net.train_reinforce(batch_states, batch_actions, advantages, batch_masks)

        win_rate = batch_wins / n_games_per_batch
        draw_rate = batch_draws / n_games_per_batch
        loss_rate = batch_losses / n_games_per_batch
        recent_win_rates.append(win_rate)
        if len(recent_win_rates) > 10:
            recent_win_rates.pop(0)
        avg_win_rate = sum(recent_win_rates) / len(recent_win_rates)

        if batch_idx % 5 == 0 or win_rate >= target_win_rate:
            print(
                f"Batch {batch_idx:3d} | Loss: {loss:7.4f} | "
                f"Win: {win_rate:5.1%} | Draw: {draw_rate:5.1%} | Loss: {loss_rate:5.1%} | "
                f"10-Batch Avg Win: {avg_win_rate:5.1%}"
            )

        if avg_win_rate >= target_win_rate and batch_idx >= 15:
            eval_win, eval_draw, eval_loss = evaluate(net, 200)
            print("\n=== Evaluation (200 games, greedy policy) ===")
            print(f"Wins: {eval_win:.1%} | Draws: {eval_draw:.1%} | Losses: {eval_loss:.1%}")
            if eval_win >= target_win_rate:
                net.save("model.npz")
                print("Target win rate achieved! Model saved to 'model.npz'.")
                break

    return net


if __name__ == "__main__":
    trained_net = train()