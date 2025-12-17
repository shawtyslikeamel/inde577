import numpy as np
import pytest

from rice_ml.unsupervised_learning.k_means_clustering import KMeans


def test_kmeans_basic_two_clusters():
    # Two very obvious clusters
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [10.0, 11.0],
        [11.0, 10.0],
    ])

    km = KMeans(n_clusters=2, max_iter=50, random_state=0)
    km.fit(X)

    # centroids should exist and have correct shape
    assert km.centroids_ is not None
    assert km.centroids_.shape == (2, 2)

    labels = km.predict(X)
    assert labels.shape == (X.shape[0],)

    # First 3 points same cluster, last 3 points same cluster
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]


def test_kmeans_reproducibility():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [10.0, 11.0],
        [11.0, 10.0],
    ])

    km1 = KMeans(n_clusters=2, random_state=123).fit(X)
    km2 = KMeans(n_clusters=2, random_state=123).fit(X)

    labels1 = km1.predict(X)
    labels2 = km2.predict(X)

    assert np.array_equal(labels1, labels2)


def test_kmeans_predict_before_fit_raises():
    X = np.array([[0.0, 0.0]])

    km = KMeans(n_clusters=1)

    with pytest.raises(Exception):
        km.predict(X)
