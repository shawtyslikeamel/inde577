import numpy as np
import pytest

from rice_ml.supervised_learning.ensemble_methods import BaggingTreeClassifier


def make_toy_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    # Simple linearly-separable-ish labels with noise
    y = (X[:, 0] + 0.25 * X[:, 1] > 0).astype(int)
    return X, y


def test_fit_predict_shapes_and_labels():
    X, y = make_toy_data()

    model = BaggingTreeClassifier(n_estimators=7, max_depth=3, random_state=123)
    model.fit(X, y)

    # predict_proba shape: (n_samples, n_classes)
    proba = model.predict_proba(X)
    assert proba.shape[0] == X.shape[0]
    assert proba.ndim == 2
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-7)

    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)
    # should predict only classes present (0/1)
    assert set(np.unique(preds)).issubset({0, 1})


def test_not_fitted_raises():
    X, _ = make_toy_data()
    model = BaggingTreeClassifier(n_estimators=3, random_state=0)

    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)

    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_proba(X)


def test_reproducibility_with_random_state():
    """Test random_state reproducibility with more features."""
    # Create dataset with MORE features to increase randomness
    rng = np.random.default_rng(0)
    n_samples = 500
    n_features = 10  
    
    X = rng.normal(size=(n_samples, n_features))
    # Make a more complex decision boundary
    y = (X[:, 0] + X[:, 1] - X[:, 2] + 0.5 * X[:, 3] > 0).astype(int)
    
    m1 = BaggingTreeClassifier(
        n_estimators=20,
        max_depth=6,
        max_features=3, 
        random_state=42
    ).fit(X, y)
    
    m2 = BaggingTreeClassifier(
        n_estimators=20,
        max_depth=6,
        max_features=3,
        random_state=42
    ).fit(X, y)
    
    m3 = BaggingTreeClassifier(
        n_estimators=20,
        max_depth=6,
        max_features=3,
        random_state=43
    ).fit(X, y)
    
    p1 = m1.predict(X)
    p2 = m2.predict(X)
    p3 = m3.predict(X)
    
    # Same seed = same predictions
    assert np.array_equal(p1, p2), "Same seed should give same predictions"
    
    # Different seed = different predictions
    n_diff = np.sum(p1 != p3)
    print(f"Predictions differ on {n_diff}/{len(X)} samples ({100*n_diff/len(X):.1f}%)")
    
    assert n_diff > len(X) * 0.01, f"Only {n_diff} predictions differ - not enough randomness!"