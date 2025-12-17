import numpy as np

from rice_ml.unsupervised_learning.pca import PCA


def test_pca_reduces_dimension():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))

    p = PCA(n_components=2)
    Z = p.fit_transform(X)

    assert Z.shape == (20, 2)
    assert p.components_.shape == (2, 5)
