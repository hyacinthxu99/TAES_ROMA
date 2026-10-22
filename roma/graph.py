"""Correlation graph from training observations."""
import numpy as np


def correlation_graph(train_data, threshold=.5):
    """Return I + D^-1/2 A D^-1/2 from [time, channels] training observations.

    A uses thresholded absolute Pearson correlations with no diagonal edges.
    A correlation graph is an association proxy, not identified causal structure.
    """
    data = np.asarray(train_data, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1:
        raise ValueError("train_data must have shape [time>=2, channels>=1]")
    if not np.isfinite(data).all() or not 0 <= threshold <= 1:
        raise ValueError("Expected finite training data and threshold in [0, 1]")
    centered = data-data.mean(axis=0)
    norm = np.linalg.norm(centered, axis=0)
    denominator = norm[:, None]*norm[None, :]
    corr = np.divide(centered.T@centered, denominator,
                     out=np.zeros_like(denominator), where=denominator>0)
    corr = np.clip(corr, -1., 1.)
    adjacency = np.where(np.abs(corr)>=threshold, np.abs(corr), 0.)
    np.fill_diagonal(adjacency, 0.)
    degree = adjacency.sum(axis=1)
    inverse = np.zeros_like(degree)
    np.power(degree, -.5, out=inverse, where=degree>0)
    return np.eye(data.shape[1]) + inverse[:, None]*adjacency*inverse[None, :]
