# Reinforce-tic-tac-toe

A neural network trained via Policy Gradient (REINFORCE) to play Tic-Tac-Toe.

## Overview

- **Algorithm**: REINFORCE (Monte-Carlo Policy Gradient) with discounted returns, baseline standardization, and illegal move masking.
- **Model**: Fully-connected network in NumPy (`9 -> 256 -> 9`) with ReLU activations, He initialization, and Adam optimization.
- **Environment**: Custom 3x3 Tic-Tac-Toe board logic with move validation and win condition detection.
- **Performance**: Consistently achieves **>90% win rate** against random opponent baseline.

## Project Structure

- `game.py`: Tic-Tac-Toe game engine, rules, state representation, win detection, and CLI board display.
- `nn.py`: Pure NumPy `Network` implementation with Adam optimizer, forward/backward passes, REINFORCE gradient updates, and model save/load (`model.npz`).
- `train.py`: Batch training loop using REINFORCE policy gradients, discounted rewards, advantage normalization, and model checkpointing.
- `play.py`: Interactive CLI to play against the trained neural network as Player 1 (X) or Player 2 (O).
- `t.py`: Masking utility script for verifying occupied cell handling.

## Quick Start

### 1. Train the Neural Network

```bash
python train.py
```
This runs batches of games, logs win/draw/loss rates, and automatically saves weights to `model.npz` once the target win rate (≥ 90%) is achieved.

### 2. Play Against the AI

```bash
python play.py
```
Choose whether to go first (Player 1 / X) or second (Player 2 / O), and enter coordinates `row col` (e.g. `1 1`).