import numpy as np

from rice_ml.unsupervised_learning.dbscan import DBSCAN


def test_dbscan_finds_clusters_and_noise():
    X = np.array([
        [0,0],[0,1],[1,0],[1,1],     # dense cluster
        [10,10],[10,11],[11,10],[11,11],  # second dense cluster
        [50,50],  # noise
    ], dtype=float)

    db = DBSCAN(eps=1.5, min_samples=3)
    labels = db.fit_predict(X)

    assert labels.shape == (len(X),)
    assert -1 in labels  # noise point
    assert len(set(labels)) >= 3  # two clusters + noise (-1)
