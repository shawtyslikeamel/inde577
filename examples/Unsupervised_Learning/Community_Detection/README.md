# Community Detection

This directory contains example code and notes for the Community Detection algorithm
in unsupervised learning.

## Algorithm

Community detection treats the dataset as a graph:
- nodes represent data points
- edges connect points based on a similarity rule (e.g., distance threshold or k-nearest graph)
A community detection algorithm then finds groups of nodes that are more densely connected to each other
than to the rest of the graph.

Key hyperparameters depend on the graph construction:
- distance threshold or number of neighbors
- similarity/distance metric

## Data

We use coordinate-based features from `lesions_processed.csv` to build a graph of lesion points.
No labels are used to construct communities, but the notebook may compare community membership
to `tampered` or `experiment` afterward to interpret structure.