# Supervised Learning Examples

Supervised learning uses labeled data `(X, y)` to learn a function that predicts `y` from `X`.

In this project, we treat tampering detection as a binary classification task:
- `y = tampered ∈ {0,1}`
- Features are derived from lesion coordinates and experiment condition.

## Shared Feature Set Used Across Most Notebooks

- `x_norm`, `y_norm`, `slice_norm` (normalized coordinates)
- `r_xy` (distance from image center in xy plane)
- `experiment` (1 = blind, 2 = open)

## Algorithms

- Logistic Regression
- KNN
- Perceptron
- Multilayer Perceptron (MLP)
- Decision Trees
- Regression Trees
- Ensemble Methods

Each subfolder contains a notebook with results and plots.
