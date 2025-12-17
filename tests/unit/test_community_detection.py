import numpy as np

from rice_ml.unsupervised_learning.community_detection import GraphCommunityDetection



def test_community_detection_returns_labels():
    # simple adjacency matrix with two obvious groups
    A = np.array([
        [0,1,1,0,0,0],
        [1,0,1,0,0,0],
        [1,1,0,0,0,0],
        [0,0,0,0,1,1],
        [0,0,0,1,0,1],
        [0,0,0,1,1,0],
    ], dtype=int)

    labels = GraphCommunityDetection(A)
    assert len(labels) == A.shape[0]
    assert np.issubdtype(np.asarray(labels).dtype, np.integer)
