import numpy as np


class LogisticRegressionGD:
    """
    Binary logistic regression trained with gradient descent.

    - X: numpy array of shape (n_samples, n_features)
    - y: numpy array of shape (n_samples,) with labels 0 or 1
    """

    def __init__(self, lr: float = 0.1, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights_ = None  # will include bias as first element

    @staticmethod
    def _sigmoid(z):
        """Numerically stable sigmoid."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Add a bias column of ones to X."""
        X = np.asarray(X, dtype=float)
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([ones, X])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model using gradient descent.
        """
        X = self._add_bias(X)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        n_samples, n_features = X.shape
        self.weights_ = np.zeros((n_features, 1), dtype=float)

        for _ in range(self.n_iters):
            # Linear combination
            logits = X @ self.weights_

            # Predicted probabilities (n_samples x 1)
            y_pred = self._sigmoid(logits)

            # Gradient of binary cross-entropy
            error = y_pred - y
            grad = (1.0 / n_samples) * (X.T @ error)

            # Gradient descent update
            self.weights_ -= self.lr * grad

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probabilities for class 0 and class 1.

        Output shape: (n_samples, 2)
        """
        X = self._add_bias(X)
        logits = X @ self.weights_
        probs_1 = self._sigmoid(logits).ravel()
        probs_0 = 1.0 - probs_1
        return np.vstack([probs_0, probs_1]).T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels (0 or 1) using given threshold.
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
