# Multi-task Learning for Insurance Premium Prediction

This repository contains the implementation for the master's thesis "Multitask Learning for Insurance Claim Modeling".

## Overview

This project implements and compares various traditional models and machine learning models for insurance premium prediction, focusing on both claim frequency and severity estimation using multi-task learning approaches.

## Models Implemented

- **GLM**: Generalized Linear Model
- **GAM**: Generalized Additive Model
- **GBM**: Gradient Boosting Machine
- **FFNN**: Feed-Forward Neural Network
- **MTNN**: Multi-Task Neural Network
- **MTMoE**: Multi-Task Mixture of Experts

## Key Features

- Cross-validation framework for model evaluation
- Feature selection and binning
- Support for both frequency and severity prediction
- Comprehensive model comparison and visualization
- Variable importance analysis and partial dependence plots

## Usage

Run cross-validation for any model:

```python
from model import MTNNPremiumModel
results = run_model_cv(MTNNPremiumModel, kwargs=model_params)
```

Generate plots and analysis using the provided Jupyter notebook `plot.ipynb`.

## Structure

- `main.py`: Cross-validation runner
- `model/`: Model implementations
  - `base.py`: Base model interface
  - `glm.py`: Generalized Linear Model
  - `gam.py`: Generalized Additive Model
  - `gbm.py`: Gradient Boosting Machine
  - `ffnn.py`: Feed-Forward Neural Network
  - `mtnn.py`: Multi-Task Neural Network
  - `mtmoenn.py`: Multi-Task Mixture of Experts
  - `nn.py`: Neural network utilities
- `plot.ipynb`: Visualization and analysis
- `preprocess.py`: Data preprocessing utilities
- `utils.py`: Evaluation metrics and utilities

## Contributors

- Siyi Li (lynniele2002@gmail.com)
