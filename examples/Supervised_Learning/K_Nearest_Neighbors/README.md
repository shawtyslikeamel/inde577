# K Nearest Neighbors

This directory contains example code and notes for the K Nearest Neighbors algorithm
in supervised learning.

## Algorithm

KNN is an instance-based method that predicts using the labels of the closest training points
to a query point. For classification, it uses majority vote among the k nearest neighbors
(optionally distance-weighted). Distances are computed using a metric such as Euclidean or Manhattan.

Key hyperparameters:
- k (number of neighbors)
- distance metric (euclidean, manhattan)
- weighting scheme (uniform or distance)

## Data

We use `lesions_processed.csv` with the binary target label `tampered` (0/1).
Features used are coordinate-based:
- x_norm, y_norm, slice_norm
- r_xy
- experiment

Because KNN is distance-based, feature scaling/normalization is important and the notebook uses
the preprocessed normalized columns.