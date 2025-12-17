import numpy as np
import pytest

from rice_ml.supervised_learning.regression_trees import RegressionTreeRegressor


def test_regression_tree_fits_simple_function():
    # y = x0 + 2*x1
    X = np.array([[0,0],[0,1],[1,0],[1,1],[2,1]], dtype=float)
    y = np.array([0,2,1,3,4], dtype=float)

    reg = RegressionTreeRegressor(max_depth=3, random_state=0)
    reg.fit(X, y)
    pred = reg.predict(X)

    assert pred.shape == y.shape
    # should fit training pretty well
    assert np.mean((pred - y) ** 2) < 0.5


def test_regression_tree_errors():
    reg = RegressionTreeRegressor(max_depth=2)
    with pytest.raises(Exception):
        reg.predict(np.array([[0.0, 0.0]]))  # predict before fit
