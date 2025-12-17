# Linear Regression

This directory contains example code and notes for the Linear Regression algorithm
in supervised learning.

## Algorithm

Linear Regression models a continuous target as ŷ = wᵀx + b by minimizing mean squared error (MSE).
Parameters can be learned via gradient descent or closed-form normal equations (depending on implementation).

Key hyperparameters (if gradient descent):
- learning rate
- number of iterations/epochs

## Data

We use `lesions_processed.csv`. Even though `tampered` is binary, it can be treated as a numeric target
(0.0 or 1.0) for regression to produce a continuous score. We can optionally threshold the score to form
a predicted class.

Features:
- x_norm, y_norm, slice_norm
- r_xy
- experiment