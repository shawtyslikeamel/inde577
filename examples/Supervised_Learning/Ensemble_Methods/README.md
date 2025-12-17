# Ensemble Methods

This directory contains example code and notes for the Ensemble Methods algorithm
in supervised learning.

## Algorithm

Ensemble methods combine multiple base models to improve stability and generalization.
Common approaches include:
- bagging (training models on bootstrap samples)
- voting/averaging predictions across models
- random feature subsampling (often paired with trees)

Key hyperparameters depend on the ensemble type:
- number of estimators (n_estimators)
- base learner settings (e.g., tree depth)
- sampling strategy (bootstrap size, feature subsampling)

## Data

We use `lesions_processed.csv` with binary label `tampered`.
Features are coordinate-based:
- x_norm, y_norm, slice_norm
- r_xy
- experiment

The notebook compares a single model vs an ensemble built from multiple base models.