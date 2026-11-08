from matplotlib.pylab import gamma
import numpy as np


def linear_kernel(X, Z):
    return X @ Z.T


def polynomial_kernel(X, Z, *, degree, gamma, coef0):
    return (gamma * (X @ Z.T) + coef0) ** degree


# aka RBF kernel
def gaussian_kernel(X, Z, *, gamma):
    x_sq = np.sum(X**2, axis=1)[:, None]
    z_sq = np.sum(Z**2, axis=1)[None, :]

    distances_sq = np.maximum(x_sq + z_sq - 2 * (X @ Z.T), 0.0)

    return np.exp(-gamma * distances_sq)


class SVM:
    def __init__(
        self,
        kernel="linear",
        C=1.0,
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        max_iter=1000,
        max_passes=10,
        random_state=42,
    ):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.max_passes = max_passes
        self.random_state = random_state

        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._alphas: np.ndarray | None = None
        self._b: float | None = None

    def _kernel_matrix(self, X, Z):
        if self.kernel == "linear":
            return linear_kernel(X, Z)

        if self.kernel == "polynomial":
            return polynomial_kernel(
                X, Z, degree=self.degree, gamma=self.gamma, coef0=self.coef0
            )

        if self.kernel == "gaussian":
            return gaussian_kernel(X, Z, gamma=self.gamma)

        raise ValueError("Unsupported kernel")
