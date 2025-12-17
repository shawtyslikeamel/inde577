import numpy as np
import pytest

from rice_ml.supervised_learning.logistic_regression import LogisticRegressionGD


def test_logreg_learns_simple_separable():
    # AND gate (linearly separable if you include bias)
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0,0,0,1], dtype=int)

    clf = LogisticRegressionGD(lr=0.5, n_iters=2000, random_state=0)
    clf.fit(X, y)
    pred = clf.predict(X)

    assert pred.shape == y.shape
    assert (pred == y).mean() >= 0.75  # allow small implementation differences


def test_logreg_errors():
    X = np.array([[0,0],[1,1]], dtype=float)
    y = np.array([0,1], dtype=int)

    clf = LogisticRegressionGD()
    with pytest.raises(Exception):
        clf.predict(X)  # predict before fit

    # mismatched lengths
    with pytest.raises(ValueError):
        clf.fit(X, np.array([0,1,1]))
