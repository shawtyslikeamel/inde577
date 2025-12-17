# Regression Trees

This directory contains example code and notes for the Regression Trees algorithm
in supervised learning.

## Algorithm

A Regression Tree is similar to a decision tree but predicts a continuous value at each leaf,
typically the mean of target values in that leaf. The tree chooses splits that reduce a regression
loss (e.g., variance / squared error).

Key hyperparameters:
- max_depth
- min_samples_split
- min_samples_leaf

## Data

We use `lesions_processed.csv`. Although `tampered` is binary (0/1), in this notebook it is treated
as a numeric target so the regression tree outputs a score in [0,1] (often near 0 or 1).
We can threshold predictions at 0.5 to convert back to a class prediction.

Features:
- x_norm, y_norm, slice_norm
- r_xy
- experiment