import numpy as np

from rice_ml.unsupervised_learning.community_detection import GraphCommunityDetection


def test_graph_community_detection_finds_two_groups():
    # Two tight clusters far apart (should be 2 communities)
    X = np.array([
        [0.00, 0.00],
        [0.05, 0.02],
        [0.02, 0.06],
        [5.00, 5.00],
        [5.04, 5.01],
        [5.01, 5.05],
    ], dtype=float)

    model = GraphCommunityDetection(eps=0.15).fit(X)
    labels = model.labels_

    assert labels is not None
    assert labels.shape == (X.shape[0],)
    assert labels.dtype == int

    # exactly 2 connected components
    assert len(np.unique(labels)) == 2

    # first 3 points are same community; last 3 points are same community
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]


def test_graph_community_detection_all_isolated_when_eps_tiny():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ], dtype=float)

    model = GraphCommunityDetection(eps=1e-12).fit(X)
    labels = model.labels_

    # No edges => each point is its own component
    assert len(np.unique(labels)) == X.shape[0]


def test_graph_community_detection_single_component_when_eps_large():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ], dtype=float)

    model = GraphCommunityDetection(eps=10.0).fit(X)
    labels = model.labels_

    # Big eps connects everything (through edges), so 1 component
    assert len(np.unique(labels)) == 1
