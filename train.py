import random
import numpy as np
from game import Game
from nn import Network, rng


def get_net_move(net, board, deterministic=False, temperature=1.0):
    """
    Selects a move for the neural network on the given board.

    Args:
        net: Network instance.
        board: 3x3 numpy board.
        deterministic: if True, always pick the argmax (greedy).
        temperature: softmax temperature for exploration (lower = greedier).
    """
    state = np.asarray(board, dtype=float).reshape(-1)
    legal_mask = (state == 0)
    if not legal_mask.any():
        return None, None

    probs = net.get_action_probs(state, legal_mask)

    # Apply temperature scaling (re-softmax the log-probs)
    if not deterministic and temperature != 1.0:
        log_p = np.log(probs + 1e-12)
        log_p /= temperature
        log_p -= np.max(log_p)
        probs = np.exp(log_p)
        probs /= probs.sum()

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


def play_game(net, explore=True, temperature=1.0):
    """
    Plays one game: Net (Player 1 / X) vs Random (Player 2 / O).

    Returns:
        transitions: list of (state, action, legal_mask) for each net move
        result: 1 (net won), -1 (net lost), 0 (draw)
    """
    g = Game()
    transitions = []
    turn = 1

    while not g.game_over():
        if turn == 1:
            state = g.board.copy().reshape(-1)
            legal_mask = (state == 0)
            pos, action = get_net_move(
                net, g.board,
                deterministic=not explore,
                temperature=temperature
            )
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
    wins = draws = losses = 0
    for _ in range(n_games):
        _, result = play_game(net, explore=False)
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1
    return wins / n_games, draws / n_games, losses / n_games


def train(net=None, n_games_per_batch=64, target_win_rate=0.90,
          gamma=0.99, max_batches=500):
    """
    Trains the network using REINFORCE (policy gradient) against a random player.

    Improvements over v1:
      - Higher discount factor (gamma=0.99): assign more credit to early moves.
      - Temperature-annealed exploration: start warm (T=1.5), anneal toward 0.5
        to shift from exploration → exploitation.
      - Learning rate cosine decay: smoothly reduces LR to avoid overshooting
        once the policy is strong.
      - Shaped rewards: win +1, draw +0.2, loss -1 (draw is slightly positive
        since X has a first-move advantage and drawing means defense held).
      - Entropy bonus in the network prevents premature collapse.
      - Larger eval set (300 games) for reliable convergence check.
    """
    if net is None:
        net = Network(lr=3e-3, entropy_coeff=0.02, grad_clip=1.0)

    initial_lr = net.lr
    initial_temp = 1.5
    final_temp   = 0.5

    print("Starting REINFORCE training (improved v2)...")
    print(f"  Batch size: {n_games_per_batch} | gamma: {gamma} | "
          f"entropy_coeff: {net.entropy_coeff} | grad_clip: {net.grad_clip}")
    print(f"  Temperature: {initial_temp} → {final_temp} | "
          f"LR: {initial_lr} (cosine decay)")

    recent_win_rates = []

    for batch_idx in range(1, max_batches + 1):
        # ── Schedule: temperature & learning rate ──
        progress = batch_idx / max_batches
        temperature = initial_temp + (final_temp - initial_temp) * progress
        cosine_lr = initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        net.set_lr(max(cosine_lr, 1e-4))  # floor at 1e-4

        # ── Play batch of games ──
        batch_states  = []
        batch_actions = []
        batch_advs    = []
        batch_masks   = []

        batch_wins = batch_draws = batch_losses = 0

        for _ in range(n_games_per_batch):
            transitions, result = play_game(net, explore=True,
                                            temperature=temperature)
            if result == 1:
                batch_wins += 1
                base_reward = 1.0
            elif result == -1:
                batch_losses += 1
                base_reward = -1.0
            else:
                batch_draws += 1
                base_reward = 0.2   # slight positive for draw (defense held)

            # Discounted returns — later moves get more credit (closer to outcome)
            T = len(transitions)
            for t, (s, a, m) in enumerate(transitions):
                g_t = base_reward * (gamma ** (T - 1 - t))
                batch_states.append(s)
                batch_actions.append(a)
                batch_masks.append(m)
                batch_advs.append(g_t)

        if not batch_states:
            continue

        states     = np.array(batch_states)
        actions    = np.array(batch_actions)
        masks      = np.array(batch_masks)
        advantages = np.array(batch_advs, dtype=float)

        # Advantage standardisation (variance reduction)
        adv_std = np.std(advantages)
        if adv_std > 1e-8:
            advantages = (advantages - np.mean(advantages)) / adv_std
        else:
            advantages = advantages - np.mean(advantages)

        # Update policy
        loss = net.train_reinforce(states, actions, advantages, masks)

        win_rate  = batch_wins   / n_games_per_batch
        draw_rate = batch_draws  / n_games_per_batch
        loss_rate = batch_losses / n_games_per_batch
        recent_win_rates.append(win_rate)
        if len(recent_win_rates) > 10:
            recent_win_rates.pop(0)
        avg_win_rate = sum(recent_win_rates) / len(recent_win_rates)

        if batch_idx % 5 == 0 or win_rate >= target_win_rate:
            print(
                f"Batch {batch_idx:3d} | Loss: {loss:7.4f} | "
                f"Win: {win_rate:5.1%} | Draw: {draw_rate:5.1%} | "
                f"Loss: {loss_rate:5.1%} | "
                f"AvgWin: {avg_win_rate:5.1%} | "
                f"T={temperature:.2f} LR={net.lr:.5f}"
            )

        if avg_win_rate >= target_win_rate and batch_idx >= 15:
            eval_win, eval_draw, eval_loss = evaluate(net, 300)
            print(f"\n=== Evaluation (300 games, greedy) ===")
            print(f"Wins: {eval_win:.1%} | Draws: {eval_draw:.1%} | "
                  f"Losses: {eval_loss:.1%}")
            if eval_win >= target_win_rate:
                net.save("model.npz")
                print("Target win rate achieved! Model saved to 'model.npz'.")
                break

    return net


if __name__ == "__main__":
    trained_net = train()