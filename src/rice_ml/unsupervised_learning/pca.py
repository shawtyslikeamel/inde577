import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA)

    Parameters
    ----------
    n_components : int
        Number of principal components to keep.
    """

    def __init__(self, n_components=2):
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        """
        Fit PCA model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=float)

        # 1. Center the data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # 2. Covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # 3. Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 4. Sort by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. Select top components
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / eigenvalues.sum()
        )

        return self

    def transform(self, X):
        """
        Project data onto principal components.
        """
        if self.components_ is None:
            raise RuntimeError("PCA not fitted. Call fit() first.")

        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
