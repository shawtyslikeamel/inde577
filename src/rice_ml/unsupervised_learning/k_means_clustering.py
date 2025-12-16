import numpy as np

class KMeans:
    """
    Simple K-Means clustering implementation using NumPy.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters (k).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None

    def fit(self, X):
        """
        Fit K-Means to the dataset X.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
            Input feature matrix.
        """
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        # Randomly pick initial centroids from data
        indices = rng.choice(len(X), self.n_clusters, replace=False)
        centroids = X[indices]

        for _ in range(self.max_iter):
            # Assign each point to nearest centroid
            distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
            labels = np.argmin(distances, axis=1)

            # Compute new centroids
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
                for k in range(self.n_clusters)
            ])

            # Check for convergence
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < self.tol:
                break

        self.centroids_ = centroids
        return self

    def predict(self, X):
        """
        Assign clusters to new samples.

        Parameters
        ----------
        X : ndarray

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        distances = np.linalg.norm(X[:, None] - self.centroids_[None, :], axis=2)
        return np.argmin(distances, axis=1)
