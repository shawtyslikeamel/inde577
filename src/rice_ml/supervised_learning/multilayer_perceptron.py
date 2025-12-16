"""
Multilayer Perceptron (MLP) - NumPy-only.

Minimal 1-hidden-layer neural network for binary classification.
- Sigmoid activations
- Binary cross-entropy loss
- Gradient descent (batch)

This matches typical "Neural Nets / backprop" class content.
"""

from __future__ import annotations
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class MLPBinaryClassifier:
    """
    1-hidden-layer MLP for binary classification.

    Parameters
    ----------
    hidden_units : int, default=8
        Number of hidden neurons.
    lr : float, default=0.1
        Learning rate.
    n_epochs : int, default=300
        Training epochs.
    random_state : int or None
        Seed.

    Attributes
    ----------
    W1, b1, W2, b2 : learned parameters
    """

    def __init__(self, hidden_units: int = 8, lr: float = 0.1, n_epochs: int = 300, random_state: int | None = None):
        self.hidden_units = hidden_units
        self.lr = lr
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError("y must be shape (n_samples, 1).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows.")

        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        h = self.hidden_units

        # small random init
        self.W1 = rng.normal(0, 0.1, size=(d, h))
        self.b1 = np.zeros((1, h))
        self.W2 = rng.normal(0, 0.1, size=(h, 1))
        self.b2 = np.zeros((1, 1))

        for _ in range(self.n_epochs):
            # forward
            Z1 = X @ self.W1 + self.b1
            A1 = _sigmoid(Z1)
            Z2 = A1 @ self.W2 + self.b2
            A2 = _sigmoid(Z2)  # predicted prob

            # binary cross-entropy grad (stable enough for small data)
            # dL/dZ2 = A2 - y
            dZ2 = (A2 - y) / n
            dW2 = A1.T @ dZ2
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dA1 = dZ2 @ self.W2.T
            dZ1 = dA1 * (A1 * (1 - A1))
            dW1 = X.T @ dZ1
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            # update
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        A1 = _sigmoid(X @ self.W1 + self.b1)
        A2 = _sigmoid(A1 @ self.W2 + self.b2)
        return A2.ravel()

    def predict(self, X, threshold: float = 0.5):
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)
