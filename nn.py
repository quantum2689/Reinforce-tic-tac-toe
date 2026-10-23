import os
import numpy as np

rng = np.random.default_rng()


class Network:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = {
            "w1": rng.standard_normal((9, 256)) * np.sqrt(2 / 9),
            "b1": np.zeros((1, 256)),
            "w2": rng.standard_normal((256, 9)) * np.sqrt(2 / 256),
            "b2": np.zeros((1, 9)),
        }
        # Adam state: first and second moment per parameter
        self.m = {k: np.zeros_like(p) for k, p in self.params.items()}
        self.v = {k: np.zeros_like(p) for k, p in self.params.items()}
        self.t = 0
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.x = None
        self.z1 = None
        self.a1 = None
        self.logits = None
        self.probs = None

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, z):
        z = np.atleast_2d(z)
        z_max = np.max(z, axis=-1, keepdims=True)
        # Avoid NaN when all values in a row are -inf
        z_max = np.where(np.isneginf(z_max), 0.0, z_max)
        e = np.exp(z - z_max)
        e_sum = np.sum(e, axis=-1, keepdims=True)
        e_sum = np.where(e_sum == 0, 1.0, e_sum)
        return e / e_sum

    def _to_onehot(self, y):
        y = np.asarray(y)
        if np.issubdtype(y.dtype, np.integer):
            # class indices: scalar, or array of N indices
            return np.eye(9)[y.reshape(-1)]    # (N,9)
        return y.reshape(-1, 9)                # already one-hot / probabilities

    def forward(self, x):
        p = self.params
        self.x = np.asarray(x, dtype=float).reshape(-1, 9)   # (N,9)
        self.z1 = self.x @ p["w1"] + p["b1"]
        self.a1 = self.relu(self.z1)
        self.logits = self.a1 @ p["w2"] + p["b2"]
        self.probs = self.softmax(self.logits)               # (N,9)
        return self.probs

    def get_action_probs(self, board, legal_mask=None):
        """Returns action probabilities over 9 cells with illegal moves masked out."""
        board_flat = np.asarray(board, dtype=float).reshape(1, 9)
        if legal_mask is None:
            legal_mask = (board_flat == 0)
        else:
            legal_mask = np.asarray(legal_mask, dtype=bool).reshape(1, 9)

        p = self.params
        z1 = board_flat @ p["w1"] + p["b1"]
        a1 = self.relu(z1)
        logits = (a1 @ p["w2"] + p["b2"]).copy()

        # Mask illegal cells
        logits[~legal_mask] = -1e9
        probs = self.softmax(logits).reshape(-1)
        return probs

    def loss(self, y):
        y = self._to_onehot(y)
        N = y.shape[0]
        return -np.sum(y * np.log(self.probs + 1e-12)) / N

    def backward_dlogits(self, dlogits):
        """Backpropagates gradient of loss with respect to logits and updates parameters with Adam."""
        p = self.params
        grads = {
            "w2": self.a1.T @ dlogits,                    # (256,9)
            "b2": dlogits.sum(axis=0, keepdims=True),     # (1,9)
        }
        da1 = dlogits @ p["w2"].T                         # (N,256)
        dz1 = da1 * (self.z1 > 0)                         # relu derivative
        grads["w1"] = self.x.T @ dz1                      # (9,256)
        grads["b1"] = dz1.sum(axis=0, keepdims=True)      # (1,256)

        self.adam_step(grads)

    def backward(self, y):
        """Supervised cross-entropy backward pass."""
        y = self._to_onehot(y)
        N = y.shape[0]
        dlogits = (self.probs - y) / N                    # (N,9)
        self.backward_dlogits(dlogits)

    def train_reinforce(self, states, actions, advantages, legal_masks=None):
        """
        Policy gradient (REINFORCE) update step.
        - states: (N, 9) array of board states
        - actions: (N,) array of chosen action indices (0..8)
        - advantages: (N,) array of scalar advantages / returns
        - legal_masks: (N, 9) boolean array of legal moves at each step
        """
        states = np.asarray(states, dtype=float).reshape(-1, 9)
        actions = np.asarray(actions, dtype=int).reshape(-1)
        advantages = np.asarray(advantages, dtype=float).reshape(-1)
        N = states.shape[0]

        if legal_masks is None:
            legal_masks = (states == 0)
        else:
            legal_masks = np.asarray(legal_masks, dtype=bool).reshape(-1, 9)

        # Forward pass on all states in batch
        self.forward(states)

        # Masked softmax over legal actions
        masked_logits = self.logits.copy()
        masked_logits[~legal_masks] = -1e9
        probs = self.softmax(masked_logits)

        # dlogits for REINFORCE loss: L = - 1/N * sum(A * log(pi(a)))
        # dL/dz = (pi - one_hot(a)) * A / N
        one_hot_actions = np.eye(9)[actions]
        dlogits = (probs - one_hot_actions) * advantages[:, None] / N
        dlogits[~legal_masks] = 0.0

        self.backward_dlogits(dlogits)

        # Compute diagnostic loss
        picked_probs = probs[np.arange(N), actions]
        loss = -np.mean(advantages * np.log(picked_probs + 1e-12))
        return float(loss)

    def adam_step(self, grads):
        self.t += 1
        for k in self.params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g * g
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)   # bias correction
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

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