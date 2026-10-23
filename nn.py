import os
import numpy as np

rng = np.random.default_rng()


class Network:
    """
    Policy network for Tic-Tac-Toe using REINFORCE.

    Architecture: Input(9) -> Dense(128, ReLU) -> Dense(128, ReLU) -> Dense(9, Softmax)

    Improvements over v1:
      - Two hidden layers (128 each) instead of one (256): deeper representation
        with fewer total parameters for better generalisation.
      - Entropy bonus in policy gradient to prevent premature collapse.
      - Gradient clipping (max-norm) for training stability.
      - Learning rate decay support.
      - He initialisation on all weight matrices.
    """

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 entropy_coeff=0.02, grad_clip=1.0):
        self.params = {
            "w1": rng.standard_normal((9, 128)) * np.sqrt(2 / 9),
            "b1": np.zeros((1, 128)),
            "w2": rng.standard_normal((128, 128)) * np.sqrt(2 / 128),
            "b2": np.zeros((1, 128)),
            "w3": rng.standard_normal((128, 9)) * np.sqrt(2 / 128),
            "b3": np.zeros((1, 9)),
        }
        # Adam state
        self.m = {k: np.zeros_like(p) for k, p in self.params.items()}
        self.v = {k: np.zeros_like(p) for k, p in self.params.items()}
        self.t = 0
        self.lr = lr
        self.initial_lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip

        # Cached activations for backward pass
        self.x = None
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        self.logits = None
        self.probs = None

    # ── Activation helpers ──────────────────────────────────────────

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, z):
        z = np.atleast_2d(z)
        z_max = np.max(z, axis=-1, keepdims=True)
        z_max = np.where(np.isneginf(z_max), 0.0, z_max)
        e = np.exp(z - z_max)
        e_sum = np.sum(e, axis=-1, keepdims=True)
        e_sum = np.where(e_sum == 0, 1.0, e_sum)
        return e / e_sum

    # ── Forward pass ────────────────────────────────────────────────

    def forward(self, x):
        p = self.params
        self.x = np.asarray(x, dtype=float).reshape(-1, 9)
        self.z1 = self.x  @ p["w1"] + p["b1"]
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ p["w2"] + p["b2"]
        self.a2 = self.relu(self.z2)
        self.logits = self.a2 @ p["w3"] + p["b3"]
        self.probs = self.softmax(self.logits)
        return self.probs

    def get_action_probs(self, board, legal_mask=None):
        """Masked action probabilities for a single board position."""
        board_flat = np.asarray(board, dtype=float).reshape(1, 9)
        if legal_mask is None:
            legal_mask = (board_flat == 0)
        else:
            legal_mask = np.asarray(legal_mask, dtype=bool).reshape(1, 9)

        p = self.params
        z1 = board_flat @ p["w1"] + p["b1"]
        a1 = self.relu(z1)
        z2 = a1 @ p["w2"] + p["b2"]
        a2 = self.relu(z2)
        logits = (a2 @ p["w3"] + p["b3"]).copy()
        logits[~legal_mask] = -1e9
        probs = self.softmax(logits).reshape(-1)
        return probs

    # ── One-hot helper ──────────────────────────────────────────────

    def _to_onehot(self, y):
        y = np.asarray(y)
        if np.issubdtype(y.dtype, np.integer):
            return np.eye(9)[y.reshape(-1)]
        return y.reshape(-1, 9)

    # ── Supervised loss / backward ──────────────────────────────────

    def loss(self, y):
        y = self._to_onehot(y)
        N = y.shape[0]
        return -np.sum(y * np.log(self.probs + 1e-12)) / N

    def backward_dlogits(self, dlogits):
        """Back-propagate through three layers and apply Adam update."""
        p = self.params
        grads = {}

        # Layer 3
        grads["w3"] = self.a2.T @ dlogits
        grads["b3"] = dlogits.sum(axis=0, keepdims=True)

        # Layer 2
        da2 = dlogits @ p["w3"].T
        dz2 = da2 * (self.z2 > 0)
        grads["w2"] = self.a1.T @ dz2
        grads["b2"] = dz2.sum(axis=0, keepdims=True)

        # Layer 1
        da1 = dz2 @ p["w2"].T
        dz1 = da1 * (self.z1 > 0)
        grads["w1"] = self.x.T @ dz1
        grads["b1"] = dz1.sum(axis=0, keepdims=True)

        # Gradient clipping
        if self.grad_clip > 0:
            total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
            if total_norm > self.grad_clip:
                scale = self.grad_clip / (total_norm + 1e-8)
                grads = {k: g * scale for k, g in grads.items()}

        self.adam_step(grads)

    def backward(self, y):
        """Supervised cross-entropy backward pass."""
        y = self._to_onehot(y)
        N = y.shape[0]
        dlogits = (self.probs - y) / N
        self.backward_dlogits(dlogits)

    # ── REINFORCE policy gradient ───────────────────────────────────

    def train_reinforce(self, states, actions, advantages, legal_masks=None):
        """
        Policy gradient (REINFORCE) update with entropy regularisation.

        Improvements:
          - Entropy bonus:  H = -sum(pi * log(pi))  added to the objective
            to prevent premature policy collapse and encourage exploration.
          - Gradient clipping inside backward_dlogits.
        """
        states     = np.asarray(states, dtype=float).reshape(-1, 9)
        actions    = np.asarray(actions, dtype=int).reshape(-1)
        advantages = np.asarray(advantages, dtype=float).reshape(-1)
        N = states.shape[0]

        if legal_masks is None:
            legal_masks = (states == 0)
        else:
            legal_masks = np.asarray(legal_masks, dtype=bool).reshape(-1, 9)

        # Forward pass
        self.forward(states)

        # Masked softmax over legal actions
        masked_logits = self.logits.copy()
        masked_logits[~legal_masks] = -1e9
        probs = self.softmax(masked_logits)

        # Policy gradient:  dL/dz = (pi - one_hot(a)) * advantage / N
        one_hot_actions = np.eye(9)[actions]
        dlogits = (probs - one_hot_actions) * advantages[:, None] / N

        # Entropy bonus gradient:  dH/dz = -pi * (1 + log(pi)) / N
        # We SUBTRACT this (maximise entropy = minimise negative entropy)
        if self.entropy_coeff > 0:
            log_probs = np.log(probs + 1e-12)
            entropy_grad = probs * (1.0 + log_probs) / N
            dlogits -= self.entropy_coeff * entropy_grad

        dlogits[~legal_masks] = 0.0

        self.backward_dlogits(dlogits)

        # Diagnostic loss
        picked_probs = probs[np.arange(N), actions]
        policy_loss  = -np.mean(advantages * np.log(picked_probs + 1e-12))
        entropy      = -np.mean(np.sum(probs * np.log(probs + 1e-12), axis=1))
        return float(policy_loss - self.entropy_coeff * entropy)

    # ── Adam optimiser ──────────────────────────────────────────────

    def adam_step(self, grads):
        self.t += 1
        for k in self.params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g * g
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def set_lr(self, new_lr):
        """Update the learning rate (e.g. for decay schedules)."""
        self.lr = new_lr

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, filepath="model.npz"):
        """Saves network weights to a .npz file."""
        np.savez(filepath, **self.params)

    def load(self, filepath="model.npz"):
        """Loads network weights from a .npz file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file '{filepath}' not found.")
        data = np.load(filepath)
        for k in self.params:
            if k in data:
                self.params[k] = data[k]


# Alias for backward compatibility
network = Network