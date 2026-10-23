import os
import numpy as np
from game import Game
from nn import Network
from train import get_net_move, train


def play_vs_ai():
    net = Network()
    model_path = "model.npz"
    if os.path.exists(model_path):
        net.load(model_path)
        print(f"Loaded trained model weights from '{model_path}'.")
    else:
        print(f"Model file '{model_path}' not found. Training a model first...")
        train(net, n_games_per_batch=64, target_win_rate=0.90)

    while True:
        print("\n" + "=" * 40)
        print("       TIC-TAC-TOE vs REINFORCE AI")
        print("=" * 40)
        choice = input("Play as (1) Player 1 (X, goes first) or (2) Player 2 (O, AI goes first)? [1/2]: ").strip()
        human_player = 1 if choice != "2" else -1
        ai_player = -human_player

        g = Game()
        turn = 1

        print(f"\nYou are {'X (Player 1)' if human_player == 1 else 'O (Player 2)'}.")
        print("Coordinates are row (0-2) and col (0-2), separated by a space (e.g. '1 1' for center).\n")

        while not g.game_over():
            print(g)
            print()

            if turn == human_player:
                while True:
                    try:
                        user_input = input("Your move [row col] (or 'q' to quit): ").strip()
                        if user_input.lower() == 'q':
                            return
                        parts = user_input.split()
                        if len(parts) != 2:
                            print("Please enter row and col separated by space, e.g. '0 2'.")
                            continue
                        pos = (int(parts[0]), int(parts[1]))
                        ok = g.player_1(pos) if human_player == 1 else g.player_2(pos)
                        if ok:
                            break
                        else:
                            print("Invalid move or cell is already taken. Try again.")
                    except (ValueError, IndexError):
                        print("Invalid input format. Use numbers 0 to 2.")
            else:
                print("AI is thinking...")
                # From AI's perspective, its pieces are 1 and opponent's pieces are -1
                board_perspective = g.board if ai_player == 1 else -g.board
                pos, _ = get_net_move(net, board_perspective, deterministic=True)
                if pos is not None:
                    if ai_player == 1:
                        g.player_1(pos)
                    else:
                        g.player_2(pos)
                    print(f"AI chose: {pos[0]} {pos[1]}")

            turn = -turn

        print("\nFinal Board:")
        print(g)
        print()
        r = g.result()
        if r == human_player:
            print("Congratulations! You won!")
        elif r == ai_player:
            print("The AI won! Better luck next time.")
        else:
            print("It's a draw!")

        again = input("\nPlay another game? (y/n): ").strip().lower()
        if again != 'y':
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    play_vs_ai()
