import numpy as np
import pytest

from rice_ml.supervised_learning.perceptron import Perceptron


def test_perceptron_linearly_separable():
    # two clusters separable by x0 + x1 > 0
    X = np.array([[-2,-1],[-1,-2],[1,2],[2,1]], dtype=float)
    y = np.array([0,0,1,1], dtype=int)

    p = Perceptron(lr=0.1, n_iters=50, random_state=0)
    p.fit(X, y)
    pred = p.predict(X)

    assert pred.tolist() == y.tolist()


def test_perceptron_errors_predict_before_fit():
    p = Perceptron()
    with pytest.raises(Exception):
        p.predict(np.array([[0.0, 0.0]]))
