import numpy as np
import pytest

from rice_ml.supervised_learning.perceptron import Perceptron


def test_perceptron_learns_linearly_separable_data():
    # Linearly separable: x0 + x1 > 0 → class 1
    X = np.array([
        [-2, -1],
        [-1, -2],
        [1,  2],
        [2,  1],
    ], dtype=float)

    y = np.array([0, 0, 1, 1], dtype=int)

    clf = Perceptron(lr=0.1, n_epochs=50, random_state=0)
    clf.fit(X, y)

    preds = clf.predict(X)

    # Should classify perfectly
    assert np.array_equal(preds, y)


def test_perceptron_predict_shape_and_values():
    X = np.array([[0, 0], [1, 1]], dtype=float)
    y = np.array([0, 1], dtype=int)

    clf = Perceptron(random_state=1)
    clf.fit(X, y)

    preds = clf.predict(X)

    assert preds.shape == (X.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})


def test_perceptron_raises_if_not_fitted():
    clf = Perceptron()
    X = np.array([[0.0, 0.0]])

    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(X)


def test_perceptron_reproducible_with_random_state():
    X = np.array([
        [-1, -1],
        [1,  1],
    ], dtype=float)
    y = np.array([0, 1], dtype=int)

    p1 = Perceptron(random_state=42).fit(X, y).predict(X)
    p2 = Perceptron(random_state=42).fit(X, y).predict(X)

    assert np.array_equal(p1, p2)
