"""
Linear Regression (NumPy-only).

Implements ordinary least squares (OLS) using the closed-form solution:
    w = (X^T X)^(-1) X^T y

This is a core regression method commonly covered early in ML courses.
"""

from __future__ import annotations
import numpy as np


class LinearRegression:
    """
    Ordinary Least Squares Linear Regression.

    Attributes
    ----------
    coef_ : np.ndarray
        Learned weights (n_features,).
    intercept_ : float
        Learned bias term.
    """

    def __init__(self, lr=None, epochs=None, random_state=None):
        self.coef_ = None
        self.intercept_ = None


    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows.")

        # Add bias column
        Xb = np.c_[np.ones(X.shape[0]), X]

        # Solve using least squares (more stable than explicit inverse)
        w, *_ = np.linalg.lstsq(Xb, y, rcond=None)

        self.intercept_ = float(w[0])
        self.coef_ = w[1:]
        return self

    def predict(self, X):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model not fitted. Call fit first.")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
