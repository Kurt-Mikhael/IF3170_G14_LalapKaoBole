from collections import OrderedDict

import numpy as np


class _KernelRowCache:
    def __init__(self, capacity, make_row):
        self.capacity, self.make_row = capacity, make_row
        self._rows = OrderedDict()

    def __len__(self):
        return len(self._rows)

    def row(self, index):
        row = self._rows.get(index)
        if row is None:
            row = self.make_row(index)
            if not np.isfinite(row).all():
                raise ValueError(
                    "kernel matrix overflowed; rescale input data or parameters"
                )
            if self.capacity:
                if len(self._rows) == self.capacity:
                    self._rows.popitem(last=False)
                self._rows[index] = row
        else:
            self._rows.move_to_end(index)
        return row


def linear_kernel(X, Z):
    return X @ Z.T


def polynomial_kernel(X, Z, *, degree, gamma, coef0):
    return (gamma * (X @ Z.T) + coef0) ** degree


def gaussian_kernel(X, Z, *, gamma):
    x_sq = np.sum(X**2, axis=1)[:, None]
    z_sq = np.sum(Z**2, axis=1)[None, :]
    return _gaussian_from_norms(X, Z, x_sq, z_sq, gamma)


def _gaussian_from_norms(X, Z, x_sq, z_sq, gamma):
    # ||x-z||^2 = ||x||^2 + ||z||^2 - 2 x.z; the norms do not depend on the pair.
    distances_sq = np.maximum(x_sq + z_sq - 2 * (X @ Z.T), 0.0)
    return np.exp(-gamma * distances_sq)
