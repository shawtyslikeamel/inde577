# K Means Clustering

This directory contains example code and notes for the K Means Clustering algorithm
in unsupervised learning.

## Algorithm

K-Means partitions data into k clusters by iteratively:
1) assigning each point to the nearest centroid
2) updating centroids as the mean of assigned points
The objective is to minimize within-cluster squared distance (inertia).

Key hyperparameters:
- k (number of clusters)
- max_iters
- tolerance / convergence rule
- random_state (initialization)

## Data

We use coordinate-based features from `lesions_processed.csv` without using labels during clustering.
Features commonly used:
- x_norm, y_norm, slice_norm
- (optionally) r_xy, experiment

After clustering, the notebook compares clusters to `tampered` and/or `experiment` to interpret results.