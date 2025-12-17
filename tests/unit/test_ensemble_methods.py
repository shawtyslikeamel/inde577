import numpy as np

from rice_ml.supervised_learning.ensemble_methods import majority_vote, average_predictions


def test_majority_vote_basic():
    preds = np.array([
        [0, 1, 1],  # sample 1 predicted by 3 models
        [1, 1, 0],
        [0, 0, 0],
    ])
    out = majority_vote(preds)
    assert out.tolist() == [1, 1, 0]


def test_average_predictions_basic():
    preds = np.array([
        [0.2, 0.6, 0.8],
        [0.0, 0.5, 1.0],
    ])
    out = average_predictions(preds)
    assert np.allclose(out, np.array([0.5333333333, 0.5]))
