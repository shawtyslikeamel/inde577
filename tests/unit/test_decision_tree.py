import numpy as np
import pytest

from rice_ml.supervised_learning.decision_trees import DecisionTreeClassifier


def test_tree_fits_xor_with_depth_2():
    # XOR needs depth 2
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0,1,1,0], dtype=int)

    tree = DecisionTreeClassifier(max_depth=2, random_state=0)
    tree.fit(X, y)
    pred = tree.predict(X)

    assert pred.tolist() == y.tolist()


def test_tree_predict_proba_shape():
    X = np.array([[0,0],[1,1],[1,0],[0,1]], dtype=float)
    y = np.array([0,1,1,0], dtype=int)

    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    proba = tree.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_tree_errors():
    tree = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        tree.predict(np.array([[0.0, 0.0]]))
