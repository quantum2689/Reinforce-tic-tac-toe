"""
Benchmark script: runs the CURRENT algorithm 5 times and reports
average batches-to-90%, final eval win rate, and wall-clock time.

This is the BEFORE snapshot — run it before applying improvements
so we have a baseline to compare against.
"""
import time, sys, contextlib, io
import numpy as np
from nn import Network
from train import play_game, evaluate, get_net_move, random_player
from game import Game

N_TRIALS = 5
TARGET   = 0.90
MAX_BATCHES = 500
BATCH_SIZE  = 64
GAMMA = 0.95


def train_once(seed=None):
    """Returns (batches_needed, eval_win, eval_draw, eval_loss, elapsed_sec)."""
    if seed is not None:
        np.random.seed(seed)
    net = Network(lr=3e-3)
    recent = []
    t0 = time.perf_counter()

    for batch_idx in range(1, MAX_BATCHES + 1):
        states, actions, advs, masks = [], [], [], []
        wins = 0
        for _ in range(BATCH_SIZE):
            transitions, result = play_game(net, explore=True)
            if result == 1:
                wins += 1
                base_reward = 1.0
            elif result == -1:
                base_reward = -1.0
            else:
                base_reward = 0.1
            T = len(transitions)
            for t, (s, a, m) in enumerate(transitions):
                states.append(s); actions.append(a); masks.append(m)
                advs.append(base_reward * (GAMMA ** (T - 1 - t)))
        if not states:
            continue
        advantages = np.array(advs, dtype=float)
        std = np.std(advantages)
        if std > 1e-8:
            advantages = (advantages - advantages.mean()) / std
        else:
            advantages -= advantages.mean()
        net.train_reinforce(np.array(states), np.array(actions),
                            advantages, np.array(masks))
        wr = wins / BATCH_SIZE
        recent.append(wr)
        if len(recent) > 10:
            recent.pop(0)
        avg_wr = sum(recent) / len(recent)
        if avg_wr >= TARGET and batch_idx >= 15:
            ew, ed, el = evaluate(net, 200)
            if ew >= TARGET:
                return batch_idx, ew, ed, el, time.perf_counter() - t0

    ew, ed, el = evaluate(net, 200)
    return MAX_BATCHES, ew, ed, el, time.perf_counter() - t0


def main():
    print("=" * 62)
    print("  BENCHMARK — current REINFORCE algorithm  (5 trials)")
    print("=" * 62)
    results = []
    for i in range(N_TRIALS):
        batches, ew, ed, el, secs = train_once(seed=42 + i)
        results.append((batches, ew, ed, el, secs))
        print(f"  Trial {i+1}: {batches:3d} batches | "
              f"eval win {ew:.1%} draw {ed:.1%} loss {el:.1%} | "
              f"{secs:.1f}s")

    avg_b  = np.mean([r[0] for r in results])
    avg_ew = np.mean([r[1] for r in results])
    avg_s  = np.mean([r[4] for r in results])
    print("-" * 62)
    print(f"  AVERAGE: {avg_b:.0f} batches to {TARGET:.0%} | "
          f"eval win {avg_ew:.1%} | {avg_s:.1f}s")
    print("=" * 62)
    return avg_b, avg_ew, avg_s


if __name__ == "__main__":
    main()
