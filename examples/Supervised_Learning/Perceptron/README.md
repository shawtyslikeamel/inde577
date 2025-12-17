# Perceptron

This directory contains example code and notes for the Perceptron algorithm
in supervised learning.

## Algorithm

The Perceptron is a linear classifier that updates weights when it makes a mistake.
Given prediction ŷ = sign(wᵀx + b), the update rule adjusts w and b toward correctly
classifying misclassified points.

Key hyperparameters:
- learning rate
- number of epochs
- shuffle/random_state (if implemented)

## Data

We use `lesions_processed.csv` with label `tampered` (0/1).
Features used:
- x_norm, y_norm, slice_norm
- r_xy
- experiment

The dataset is split into train/test before training the perceptron.