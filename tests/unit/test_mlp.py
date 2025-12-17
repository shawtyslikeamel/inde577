import numpy as np

from rice_ml.supervised_learning.multilayer_perceptron import MLPClassifier


def test_mlp_shapes():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0,1,1,0], dtype=int)

    mlp = MLPClassifier(hidden_layer_sizes=(4,), lr=0.1, epochs=200, random_state=0)
    mlp.fit(X, y)
    pred = mlp.predict(X)

    assert pred.shape == y.shape
    assert set(np.unique(pred)).issubset({0,1})
