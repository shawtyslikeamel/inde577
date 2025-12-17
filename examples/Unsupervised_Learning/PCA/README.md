# PCA

This directory contains example code and notes for the PCA algorithm
in unsupervised learning.

## Algorithm

PCA is a linear dimensionality reduction method that projects data onto directions (principal components)
that capture maximum variance. It is computed from the covariance matrix via eigen-decomposition or SVD.

Key hyperparameters:
- number of components (k)

## Data

We use features from `lesions_processed.csv`, such as:
- x_norm, y_norm, slice_norm, r_xy, experiment

The notebook projects the data into 2D (PC1, PC2) and visualizes structure, often coloring by
`experiment` or `tampered` for interpretation (labels are not used to fit PCA).