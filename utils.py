import numpy as np
import scipy
import torch
import pandas as pd
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance

def cat_to_code(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.cat.codes if col.dtype.name == 'category' else col)

def cat_to_dummies(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=df.select_dtypes(include=['category']).columns.tolist(), drop_first=True)

def generate_X_grid(X, n_grid_points=100):
    """
    Generate a grid of X values for prediction/plotting

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input features
    n_grid_points : int, default=100
        Number of grid points per feature

    Returns:
    --------
    X_grid : array, shape (n_grid_points**n_features, n_features)
        Grid of X values
    """
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    n_features = X.shape[1]
    grids = []

    for i in range(n_features):
        feature_min = X[:, i].min()
        feature_max = X[:, i].max()
        grid = np.linspace(feature_min, feature_max, n_grid_points)
        grids.append(grid)

    # Create meshgrid for all combinations
    if n_features == 1:
        return grids[0].reshape(-1, 1)
    else:
        mesh = np.meshgrid(*grids)
        X_grid = np.column_stack([m.ravel() for m in mesh])
        return X_grid

def mean_poisson_loglike_full(y_true, y_pred, sample_weight=None):
    if sample_weight is not None:
        sample_weight = np.ones_like(y_pred)
    y_pred = y_pred * sample_weight
    return np.average(scipy.special.xlogy(y_true, y_pred) - y_pred - scipy.special.gammaln(y_true + 1))


def mean_gamma_loglike_full(y_true, y_pred, sample_weight=None, scale=1):
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)
    alpha = sample_weight / scale
    theta = y_pred / alpha
    return scipy.stats.gamma.logpdf(y_true, a=alpha, scale=theta).mean()

__all__ = [
    "generate_X_grid",
    "mean_poisson_deviance",
    "mean_gamma_deviance",
    "mean_poisson_loglike_full",
    "mean_gamma_loglike_full",
]

