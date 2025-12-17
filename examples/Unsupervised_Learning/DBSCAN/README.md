# DBSCAN

This directory contains example code and notes for the DBSCAN algorithm
in unsupervised learning.

## Algorithm

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups points that are in dense regions.
A point is a “core” point if it has at least `min_samples` points within distance `eps`.
Clusters grow by connecting density-reachable points, and points not belonging to any cluster are labeled as noise (-1).

Key hyperparameters:
- eps (neighborhood radius)
- min_samples (minimum points to form a dense region)

## Data

We use coordinate-based features from `lesions_processed.csv` for clustering (no labels used during training).
Features used typically include:
- x_norm, y_norm, slice_norm
- (optional) r_xy

After clustering, the notebook visualizes clusters and optionally compares them to `tampered`.