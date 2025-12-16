"""
Perceptron (single-layer) classifier.

Implements the classic perceptron learning rule:
- Binary classification with labels {0,1}
- Uses a step activation on a linear score

This is a core class concept and a precursor to MLPs.
"""

from __future__ import annotations
import numpy as np


class Perceptron:
    """
    Perceptron binary classifier.

    Parameters
    ----------
    lr : float, default=0.1
        Learning rate.
    n_epochs : int, default=50
        Number of passes over the training data.
    shuffle : bool, default=True
        Whether to shuffle data each epoch.
    random_state : int or None
        Seed.

    Attributes
    ----------
    w_ : np.ndarray
        Weight vector.
    b_ : float
        Bias term.
    """

    def __init__(self, lr: float = 0.1, n_epochs: int = 50, shuffle: bool = True, random_state: int | None = None):
        self.lr = lr
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.random_state = random_state
        self.w_ = None
        self.b_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows.")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Perceptron expects binary labels encoded as 0/1.")

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.w_ = np.zeros(n_features, dtype=float)
        self.b_ = 0.0

        idx = np.arange(n_samples)

        for _ in range(self.n_epochs):
            if self.shuffle:
                rng.shuffle(idx)

            for i in idx:
                xi = X[i]
                yi = y[i]
                yhat = self._predict_one(xi)

                # update if misclassified
                update = self.lr * (yi - yhat)
                if update != 0.0:
                    self.w_ += update * xi
                    self.b_ += update

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("Model not fitted. Call fit first.")
        X = np.asarray(X, dtype=float)
        return X @ self.w_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return (scores >= 0.0).astype(int)

    def _predict_one(self, x: np.ndarray) -> int:
        score = float(x @ self.w_ + self.b_)
        return 1 if score >= 0.0 else 0
