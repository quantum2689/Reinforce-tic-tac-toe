# Reinforce-tic-tac-toe

A neural network trained via Policy Gradient (REINFORCE) to play Tic-Tac-Toe, featuring an interactive Flask Web Dashboard with live visual training, self-play, and human-in-the-loop online learning.

## Overview

- **Algorithm**: REINFORCE (Monte-Carlo Policy Gradient) with discounted returns, baseline standardization, and illegal move masking.
- **Model**: Fully-connected network in NumPy (`9 -> 256 -> 9`) with ReLU activations, He initialization, and Adam optimization.
- **Environment**: Custom 3x3 Tic-Tac-Toe board logic with move validation and win condition detection.
- **Performance**: Consistently achieves **>90% win rate** against random opponent baseline.

---

## 🌟 Interactive Flask Web Dashboard

Launch the web app:

```bash
python app.py
```
Open your browser at **`http://localhost:5000`**.

### The 4 Interactive Modes:

1. 🤖 **Watch Model Learn (vs Baseline)**
   - Watch the neural net learn against a random player move-by-move.
   - Live probability heatmaps on empty squares show where the model wants to play.
   - Interactive charts plot Win Rate and Loss over time.
   - Control playback speed, pause, or run +20 turbo games.

2. 🔄 **Watch Self-Play Training**
   - The model plays as both Player 1 (X) and Player 2 (O).
   - Watch both sides discover offensive tactics and defensive counters.
   - Chart tracks Draw Rate (defense quality), X-Wins, and O-Wins.

3. ⚔️ **Play vs Saved Model**
   - Challenge the saved model (`model.npz`).
   - Play as Player 1 (X, goes first) or Player 2 (O, goes second).
   - View the AI's move probabilities on each cell as you decide your moves.

4. 🧠 **Play & Teach Model Live (Online Learning)**
   - Play against an active neural net that learns from YOU!
   - Every game you complete triggers an immediate policy gradient update.
   - Real-time "Brain Update" card displays assigned reward, backprop loss, and behavioral adaptation.
   - Start from scratch or fine-tune the saved model.

---

## Command Line Scripts

### Train in Terminal
```bash
python train.py
```
Trains until target win rate (≥ 90%) is reached and saves weights to `model.npz`.

### Play in Terminal
```bash
python play.py
```
Interactive CLI match against the saved model.

### Run Unit Tests
```bash
python -m unittest test_game_and_nn.py test_app.py
```