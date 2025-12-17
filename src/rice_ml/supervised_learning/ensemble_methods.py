"""
Ensemble methods (NumPy-only): Bagging / Random-Forest-style classifier.

This module implements a simple ensemble of decision trees trained on
bootstrapped samples. Predictions are aggregated by majority vote.

Why this fits class scope:
- Uses decision trees (already in your repo)
- Demonstrates bagging + variance reduction
- No sklearn, pure NumPy + your own code
"""

from __future__ import annotations
import numpy as np

from .decision_trees import DecisionTreeClassifier


class BaggingTreeClassifier:
    """
    Bagging ensemble of DecisionTreeClassifier.

    Parameters
    ----------
    n_estimators : int, default=25
        Number of trees in the ensemble.
    max_depth : int or None, default=None
        Max depth for each tree.
    max_features : int or float or None, default=None
        Passed to DecisionTreeClassifier to subsample features (Random-Forest-style).
        - None: use all features
        - int: consider that many features at each split
        - float in (0,1]: fraction of features
    random_state : int or None
        Seed for reproducibility.

    Attributes
    ----------
    estimators_ : list
        List of fitted DecisionTreeClassifier objects.
    rng_ : np.random.Generator
        Random number generator instance (stored for reproducibility).
    """

    def __init__(
        self,
        n_estimators: int = 25,
        *,
        max_depth: int | None = None,
        max_features: int | float | None = None,
        random_state: int | None = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_: list[DecisionTreeClassifier] = []
        self.rng_ = np.random.default_rng(self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        #Reset RNG before fitting (ensures reproducibility on refit)
        self.rng_ = np.random.default_rng(self.random_state)
        
        n = X.shape[0]
        self.estimators_ = []

        for _ in range(self.n_estimators):
            # Bootstrap sample - use self.rng_ not a new generator
            idx = self.rng_.integers(0, n, size=n)
            
            d = X.shape[1]
            max_feats = self.max_features
            if max_feats is None:
                max_feats = 1  # ensures feature subsampling even when d=2

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=max_feats,
                #Use self.rng_ to generate child seeds
                random_state=int(self.rng_.integers(0, 1_000_000_000)),
            )

            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(self.estimators_) == 0:
            raise RuntimeError("Model not fitted. Call fit first.")
        X = np.asarray(X, dtype=float)

        # average probabilities across trees
        probas = [est.predict_proba(X) for est in self.estimators_]
        return np.mean(probas, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)