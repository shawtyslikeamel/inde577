import numpy as np

from rice_ml.supervised_learning.linear_regression import LinearRegression


def test_linear_regression_fits_line():
    X = np.array([[0],[1],[2],[3]], dtype=float)
    y = np.array([1,3,5,7], dtype=float)  # y = 2x + 1

    lr = LinearRegression(lr=0.1, epochs=2000, random_state=0)
    lr.fit(X, y)
    pred = lr.predict(X)

    mse = np.mean((pred - y)**2)
    assert mse < 1e-2
