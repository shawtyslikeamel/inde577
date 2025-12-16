from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class _RTNode:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_RTNode"] = None
    right: Optional["_RTNode"] = None
    value: Optional[float] = None  # leaf prediction (mean)

    def is_leaf(self) -> bool:
        return self.feature_index is None

class RegressionTreeRegressor:
    """
    CART-style Regression Tree (NumPy-only) using MSE (variance reduction).

    Parameters
    ----------
    max_depth : int | None
    min_samples_split : int
    min_samples_leaf : int
    random_state : int | None
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.tree_: Optional[_RTNode] = None
        self._rng: Optional[np.random.Generator] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionTreeRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples.")
        self._rng = np.random.default_rng(self.random_state)
        self.tree_ = self._grow(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise RuntimeError("Model not fitted. Call fit first.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        return np.array([self._traverse(row, self.tree_) for row in X], dtype=float)

    # ---------------- internal ----------------

    def _mse(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        mu = y.mean()
        return float(np.mean((y - mu) ** 2))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], np.ndarray, np.ndarray]:
        n, d = X.shape
        if n < 2 * self.min_samples_leaf:
            return None, None, np.array([], dtype=bool), np.array([], dtype=bool)

        best_feat, best_thr = None, None
        best_score = np.inf
        best_left = np.array([], dtype=bool)
        best_right = np.array([], dtype=bool)

        parent = self._mse(y)

        for j in range(d):
            col = X[:, j]
            thresholds = np.unique(col)
            if thresholds.size == 1:
                continue
            for thr in thresholds:
                left = col <= thr
                right = ~left
                nl, nr = left.sum(), right.sum()
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                score = (nl / n) * self._mse(y[left]) + (nr / n) * self._mse(y[right])
                # require improvement
                if score < best_score and score <= parent:
                    best_score = score
                    best_feat = j
                    best_thr = float(thr)
                    best_left, best_right = left, right

        return best_feat, best_thr, best_left, best_right

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> _RTNode:
        n = y.size
        leaf_value = float(y.mean())

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n < self.min_samples_split
            or np.allclose(y, y[0])
        ):
            return _RTNode(value=leaf_value)

        feat, thr, left, right = self._best_split(X, y)
        if feat is None:
            return _RTNode(value=leaf_value)

        return _RTNode(
            feature_index=feat,
            threshold=thr,
            left=self._grow(X[left], y[left], depth + 1),
            right=self._grow(X[right], y[right], depth + 1),
            value=leaf_value,
        )

    def _traverse(self, x: np.ndarray, node: _RTNode) -> float:
        while not node.is_leaf():
            assert node.feature_index is not None and node.threshold is not None
            if x[node.feature_index] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        assert node.value is not None
        return float(node.value)
