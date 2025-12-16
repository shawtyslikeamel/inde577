"""
Community detection using graph connectivity.

We build an undirected graph where nodes are data points and edges connect
points within a distance threshold. Communities are found as connected
components of this graph.

NumPy-only + basic graph traversal (DFS).
"""

from __future__ import annotations
import numpy as np


class GraphCommunityDetection:
    """
    Community detection via connected components.

    Parameters
    ----------
    eps : float
        Distance threshold for connecting nodes.
    """

    def __init__(self, eps: float = 0.2):
        self.eps = eps
        self.labels_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # adjacency list
        adj = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(X[i] - X[j]) <= self.eps:
                    adj[i].append(j)
                    adj[j].append(i)

        labels = -np.ones(n, dtype=int)
        visited = np.zeros(n, dtype=bool)
        community_id = 0

        for i in range(n):
            if not visited[i]:
                self._dfs(i, adj, visited, labels, community_id)
                community_id += 1

        self.labels_ = labels
        return self

    def _dfs(self, start, adj, visited, labels, community_id):
        stack = [start]
        visited[start] = True
        labels[start] = community_id

        while stack:
            node = stack.pop()
            for neigh in adj[node]:
                if not visited[neigh]:
                    visited[neigh] = True
                    labels[neigh] = community_id
                    stack.append(neigh)
