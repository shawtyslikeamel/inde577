# Multilayer Perceptron

This directory contains example code and notes for the Multilayer Perceptron algorithm
in supervised learning.

## Algorithm

An MLP is a feedforward neural network with one or more hidden layers. It learns nonlinear
decision boundaries by composing linear layers with nonlinear activation functions.
Training uses backpropagation and gradient descent to minimize a loss (classification loss for labels).

Key hyperparameters:
- hidden layer sizes
- learning rate
- number of epochs
- activation function (if implemented)
- batch size (if implemented)
- regularization (if implemented)

## Data

We use `lesions_processed.csv` with binary label `tampered`.
Input features:
- x_norm, y_norm, slice_norm
- r_xy
- experiment

The notebook loads data, splits train/test, trains the MLP, and evaluates performance.