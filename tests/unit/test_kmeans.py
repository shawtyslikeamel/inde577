import numpy as np

from rice_ml.unsupervised_learning.k_means_clustering import KMeans


def test_kmeans_basic_two_clusters():
    X = np.array([
        [0,0],[0,1],[1,0],
        [10,10],[10,11],[11,10]
    ], dtype=float)

    km = KMeans(n_clusters=2, max_iters=50, random_state=0)
    km.fit(X)
    labels = km.predict(X)

    assert labels.shape == (len(X),)
    assert set(np.unique(labels)) == {0,1}
    assert km.centroids_.shape == (2, 2)
