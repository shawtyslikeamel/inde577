# Decision Trees

This directory contains example code and notes for the Decision Trees algorithm
in supervised learning.

## Algorithm

A Decision Tree classifier recursively splits the feature space using thresholds on features
to reduce impurity (here, Gini impurity). Each internal node chooses a feature and threshold;
each leaf outputs class probabilities based on the training labels in that region.

Key hyperparameters:
- max_depth
- min_samples_split
- min_samples_leaf
- max_features (if implemented)

## Data

We use `lesions_processed.csv` with binary label `tampered`.
Features used include:
- x_norm, y_norm, slice_norm
- r_xy
- experiment

The notebook loads the dataset, splits into train/test, fits the tree, and evaluates predictions.