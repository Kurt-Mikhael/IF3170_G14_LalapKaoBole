import numpy as np


def linear_kernel(X, Z):
    return X @ Z.T


def polynomial_kernel(X, Z, *, degree, gamma, coef0):
    return (gamma * (X @ Z.T) + coef0) ** degree


def gaussian_kernel(X, Z, *, gamma):
    x_sq = np.sum(X**2, axis=1)[:, None]
    z_sq = np.sum(Z**2, axis=1)[None, :]
    distances_sq = np.maximum(x_sq + z_sq - 2 * (X @ Z.T), 0.0)
    return np.exp(-gamma * distances_sq)
