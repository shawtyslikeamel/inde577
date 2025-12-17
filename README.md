# CMOR 438 Data Science & Machine Learning FINAL PROJECT

Project Introduction:

This project uses metadata from the UCI Deepfakes Medical Image Tamper Detection dataset to predict whether a medical image is tampered (fake) or authentic (real). Instead of using deep learning or image processing, the model relies on tabular metadata describing each image, such as tampering type, region size, mask statistics, and image-level properties. Using Linear Regression, Logistic Regression, K-Nearest Neighbors, and K-Means Clustering implemented from scratch, we demonstrate that simple ML models can reliably identify manipulated medical images.

This repo contains a small machine learning library implemented from scratch using **NumPy only**, plus example notebooks applying each algorithm to a real-world inspired dataset: **medical CT scan tampering labels**.

## Project Context (Data Story)

This project uses labels from the “Deepfakes: Medical Image Tamper Detection” setting (CT scans). Each scan includes one or more lesion locations given as voxel coordinates `(x, y, slice)` and a ground-truth label describing whether the location is real or tampered:

- **TM (True-Malicious):** real cancer present
- **TB (True-Benign):** no cancer present
- **FM (False-Malicious):** fake cancer injected (tampered)
- **FB (False-Benign):** real cancer removed (tampered)

We combine experiments:
- **Experiment 1 (blind trial):** radiologists not told scans were tampered
- **Experiment 2 (open trial):** radiologists told scans were tampered and asked to detect

In our ML framing, we simplify to a binary label:
- `tampered = 1` for **FB or FM**
- `tampered = 0` for **TB or TM**

> Note: this repo uses the **label tables + coordinates**, not full CT images. That makes the dataset lightweight and reproducible, but also a limitation (we’re not learning pixel-level tampering).

## Repository Structure

- `src/rice_ml/` — ML algorithms from scratch (NumPy-only)
- `data/` — processed dataset CSV used by notebooks (ex: `lesions_processed.csv`)
- `examples/` — one folder per algorithm, each with a notebook + README
- `tests/` — unit tests for core library pieces (optional depending on grading)

## Algorithms Demonstrated
### Supervised Learning

- Logistic Regression (GD)
- KNN (Classifier)
- Perceptron
- Multilayer Perceptron (MLP)
- Decision Trees (Classifier)
- Regression Trees (Regressor used as a classifier via threshold)
- Ensemble Methods (tree ensembles / voting)

### Unsupervised Learning
- K-Means clustering
- DBSCAN clustering
- PCA
- Community Detection (graph-based clustering)

## Dataset Used in Notebooks

Primary file:

data/lesions_processed.csv

Can be found here: https://archive.ics.uci.edu/dataset/520/deepfakes+medical+image+tamper+detection 

## Expected columns (used across notebooks):

- x, y, slice (raw coordinates)
- x_norm, y_norm, slice_norm (normalized coordinates)
- r_xy (radius from center in xy-plane, derived feature)
- experiment (1 or 2)
- type (TB/TM/FB/FM)
- tampered (0/1)

## Academic Note

This is a course project for CMOR 438 / INDE 577 at Rice University.
All implementations are for learning and do not represent clinical tools.

### Quick Install 
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"