# Logistic Regression

This directory contains example code and notes for the Logistic Regression algorithm
in supervised learning.

## Algorithm

Logistic Regression is a linear classifier that models the probability of the positive class
using the sigmoid function:  p(y=1|x)=σ(wᵀx+b). Training minimizes a loss based on the
log-likelihood (cross-entropy), typically using gradient-based optimization.

Key hyperparameters:
- learning rate (step size for gradient descent)
- number of iterations/epochs
- regularization strength (if implemented)

## Data

We use `lesions_processed.csv`, where each row is a lesion coordinate observation from a CT scan
labeling experiment. The target label is `tampered` (0 = real, 1 = tampered).

Input features used in the notebook include normalized coordinates and derived features such as:
- x_norm, y_norm, slice_norm
- r_xy (radius in the x-y plane)
- experiment (1 or 2)

Data is loaded with pandas and split into train/test before fitting.