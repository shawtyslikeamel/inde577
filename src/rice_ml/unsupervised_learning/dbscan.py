import numpy as np

class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

    Parameters
    ----------
    eps : float
        Maximum distance between two samples to be considered neighbors.
    min_samples : int
        Minimum number of points required to form a dense region.
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        labels = np.full(n, -1)  # -1 = noise
        visited = np.zeros(n, dtype=bool)

        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                labels[i] = -1  # noise
            else:
                self._expand_cluster(X, labels, visited, i, neighbors, cluster_id)
                cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def _region_query(self, X, idx):
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, labels, visited, idx, neighbors, cluster_id):
        labels[idx] = cluster_id
        i = 0

        while i < len(neighbors):
            point = neighbors[i]

            if not visited[point]:
                visited[point] = True
                new_neighbors = self._region_query(X, point)

                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.unique(
                        np.concatenate((neighbors, new_neighbors))
                    )

            if labels[point] == -1:
                labels[point] = cluster_id

            i += 1
