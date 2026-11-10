import pickle
import warnings
from math import isfinite
from numbers import Real
from pathlib import Path

import numpy as np

from .kernel import gaussian_kernel, linear_kernel, polynomial_kernel


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
        solver="smo",
    ):
        self.kernel, self.C, self.degree, self.gamma = kernel, C, degree, gamma
        self.coef0, self.tol, self.max_iter = coef0, tol, max_iter
        self.max_passes, self.random_state, self.solver = (
            max_passes,
            random_state,
            solver,
        )

    def get_params(self, deep=True):
        return {
            name: getattr(self, name)
            for name in (
                "kernel",
                "C",
                "degree",
                "gamma",
                "coef0",
                "tol",
                "max_iter",
                "max_passes",
                "random_state",
                "solver",
            )
        }

    def set_params(self, **params):
        unknown = set(params) - set(self.get_params())
        if unknown:
            raise ValueError(f"Invalid parameter(s): {', '.join(sorted(unknown))}")
        for key, value in params.items():
            setattr(self, key, value)
        self._clear_fitted()
        return self

    def _clear_fitted(self):
        for name in (
            "_X_train",
            "_y_train",
            "_alphas",
            "_b",
            "classes_",
            "support_",
            "support_vectors_",
            "n_support_",
            "dual_coef_",
            "intercept_",
            "n_features_in_",
            "coef_",
            "n_iter_",
            "converged_",
            "fit_status_",
            "_gamma",
        ):
            self.__dict__.pop(name, None)

    @staticmethod
    def _features(X):
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must be a finite numeric 2D array") from exc
        if X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite 2D array")
        return X

    def _validate_params(self):
        if self.kernel not in ("linear", "polynomial", "poly", "gaussian", "rbf"):
            raise ValueError("kernel must be linear, polynomial/poly, or gaussian/rbf")
        if self.solver not in ("qp", "smo"):
            raise ValueError("solver must be 'qp' or 'smo'")
        for name in ("C", "tol"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        for name in ("max_iter", "max_passes"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.degree, bool)
            or not isinstance(self.degree, (int, np.integer))
            or self.degree < 1
        ):
            raise ValueError("degree must be a positive integer")
        if isinstance(self.gamma, str):
            if self.gamma not in ("scale", "auto"):
                raise ValueError(
                    "gamma must be 'scale', 'auto', or a positive finite number"
                )
        elif (
            isinstance(self.gamma, bool)
            or not isinstance(self.gamma, Real)
            or not np.isfinite(self.gamma)
            or self.gamma <= 0
        ):
            raise ValueError(
                "gamma must be 'scale', 'auto', or a positive finite number"
            )
        if (
            isinstance(self.coef0, bool)
            or not isinstance(self.coef0, Real)
            or not np.isfinite(self.coef0)
            or self.coef0 < 0
        ):
            raise ValueError("coef0 must be a nonnegative finite number")

    def fit(self, X, y):
        self._clear_fitted()
        self._validate_params()
        rng = np.random.default_rng(self.random_state)
        X = self._features(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional and match X samples")
        if y.dtype.kind not in "biufUS" or (
            y.dtype.kind in "iuf" and not np.isfinite(y).all()
        ):
            raise ValueError("y labels must be finite numbers or strings")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("SVM supports exactly two target classes")
        signs = np.where(y == classes[1], 1.0, -1.0)
        if self.gamma == "scale":
            gamma = 1.0
            if X.var() > 0:
                gamma = 1.0 / (X.shape[1] * X.var())
        elif self.gamma == "auto":
            gamma = 1.0 / X.shape[1]
        else:
            gamma = float(self.gamma)
        if self.kernel in ("polynomial", "poly"):
            K = polynomial_kernel(
                X, X, degree=self.degree, gamma=gamma, coef0=self.coef0
            )
        elif self.kernel in ("gaussian", "rbf"):
            K = gaussian_kernel(X, X, gamma=gamma)
        else:
            K = linear_kernel(X, X)
        if not np.isfinite(K).all():
            raise ValueError(
                "kernel matrix overflowed; rescale input data or parameters"
            )
        self._X_train, self._y_train, self.classes_ = X.copy(), signs, classes
        self.n_features_in_ = X.shape[1]
        self._gamma = gamma
        self._alphas = np.zeros(len(X))
        self._b = 0.0
        try:
            if self.solver == "smo":
                iterations, converged = self._fit_smo(K, rng)
            else:
                iterations, converged = self._fit_qp(K)
                self._b = self._bias(K)
        except Exception:
            self._clear_fitted()
            raise
        if not converged:
            warnings.warn("SVM optimization stopped before convergence", RuntimeWarning)
        support = np.flatnonzero(self._alphas > 0)
        self.support_ = support
        self.support_vectors_ = X[support].copy()
        self.n_support_ = np.array(
            [np.sum((self._alphas > 0) & (signs == s)) for s in (-1, 1)]
        )
        self.dual_coef_ = (self._alphas[support] * signs[support])[None, :]
        self.intercept_ = np.array([self._b])
        if self.kernel == "linear":
            self.coef_ = (self._alphas * signs) @ X
            self.coef_ = self.coef_[None, :]
        self.n_iter_, self.converged_, self.fit_status_ = (
            iterations,
            bool(converged),
            0 if converged else 1,
        )
        return self

    def _fit_qp(self, K):
        # Q weights each kernel entry by both labels; grad is Q alpha - 1.
        y, a, C = self._y_train, self._alphas, float(self.C)
        Q = (y[:, None] * y[None, :]) * K
        lipschitz = max(float(np.linalg.eigvalsh(Q)[-1]), 1e-12)
        step = 1.0 / lipschitz
        z, momentum = a.copy(), 1.0
        iteration = 0
        for iteration in range(1, self.max_iter + 1):
            candidate = self._project(z - step * (Q @ z - 1.0), y, C)
            # The projected-gradient mapping is a scaled KKT residual.
            residual = np.max(np.abs(candidate - z)) / step
            if not np.isfinite(residual):
                raise FloatingPointError("non-finite QP optimization step")
            if residual <= self.tol:
                self._alphas = candidate
                self._b = self._bias(K)
                if self._kkt_residual(K) <= self.tol:
                    a = candidate
                    break
            # Momentum extrapolates the last feasible step; project again next round.
            next_momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
            z = candidate + ((momentum - 1) / next_momentum) * (candidate - a)
            a, momentum = candidate, next_momentum
        self._alphas = a
        self._b = self._bias(K)
        return iteration, self._kkt_residual(K) <= self.tol

    @staticmethod
    def _project(v, y, C):
        # Projection clips v - lambda*y; bisection chooses lambda for y.alpha=0.
        low, high = -2 * (np.max(np.abs(v)) + C + 1), 2 * (np.max(np.abs(v)) + C + 1)
        for _ in range(80):
            mid = (low + high) / 2
            total = np.dot(y, np.clip(v - mid * y, 0, C))
            if total > 0:
                low = mid
            else:
                high = mid
        return np.clip(v - ((low + high) / 2) * y, 0, C)

    def _fit_smo(self, K, rng):
        y, a, C = self._y_train, self._alphas, float(self.C)
        bound_eps = min(1e-10, C * 1e-8)
        errors = -y.copy()
        examine_all, passes, iterations = True, 0, 0
        n = len(y)

        def take(i, j):
            if i == j:
                return False
            ai, aj, yi, yj, Ei, Ej = a[i], a[j], y[i], y[j], errors[i], errors[j]
            s = yi * yj
            # L/H bound the second alpha while preserving the label-weighted sum.
            if yi != yj:
                L, H = max(0.0, aj - ai), min(C, C + aj - ai)
            else:
                L, H = max(0.0, ai + aj - C), min(C, ai + aj)
            if L >= H:
                return False
            k11, k12, k22 = K[i, i], K[i, j], K[j, j]
            eta = k11 + k22 - 2 * k12  # Pair-direction objective curvature.
            if eta > 0:
                new_j = np.clip(aj + yj * (Ei - Ej) / eta, L, H)
            else:
                # Compare Platt's dual objective at both feasible segment endpoints.
                # Our prediction adds b; Platt's threshold has the opposite sign.
                fi = yi * (Ei - self._b) - ai * k11 - s * aj * k12
                fj = yj * (Ej - self._b) - s * ai * k12 - aj * k22
                li = ai + s * (aj - L)
                hi = ai + s * (aj - H)
                obj_l = (
                    li * fi
                    + L * fj
                    + 0.5 * li * li * k11
                    + 0.5 * L * L * k22
                    + s * L * li * k12
                )
                obj_h = (
                    hi * fi
                    + H * fj
                    + 0.5 * hi * hi * k11
                    + 0.5 * H * H * k22
                    + s * H * hi * k12
                )
                objective_eps = np.finfo(float).eps * max(C, abs(obj_l), abs(obj_h))
                if obj_l < obj_h - objective_eps:
                    new_j = L
                elif obj_h < obj_l - objective_eps:
                    new_j = H
                else:
                    new_j = aj
            if abs(new_j - aj) < 1e-8 * (new_j + aj + min(C, 1e-8)):
                return False
            new_i = ai + s * (aj - new_j)
            old_b = self._b
            # b1/b2 each place its updated free point exactly on the margin.
            b1 = old_b - Ei - yi * (new_i - ai) * k11 - yj * (new_j - aj) * k12
            b2 = old_b - Ej - yi * (new_i - ai) * k12 - yj * (new_j - aj) * k22
            if bound_eps < new_i < C - bound_eps:
                self._b = b1
            elif bound_eps < new_j < C - bound_eps:
                self._b = b2
            else:
                self._b = (b1 + b2) / 2
            a[i], a[j] = new_i, new_j
            errors[:] += (
                yi * (new_i - ai) * K[i] + yj * (new_j - aj) * K[j] + (self._b - old_b)
            )
            return True

        def cyclic(indices):
            if len(indices):
                start = int(rng.integers(len(indices)))
                for offset in range(len(indices)):
                    yield indices[(start + offset) % len(indices)]

        for iteration in range(1, self.max_iter + 1):
            indices = (
                np.arange(n)
                if examine_all
                else np.flatnonzero((a > bound_eps) & (a < C - bound_eps))
            )
            changed = 0
            for j in indices:
                r = errors[j] * y[j]
                if not ((r < -self.tol and a[j] < C) or (r > self.tol and a[j] > 0)):
                    continue
                nonbound = np.flatnonzero((a > bound_eps) & (a < C - bound_eps))
                # First choose the non-bound error furthest from E_j, then randomized scans.
                if len(nonbound) > 1:
                    first = nonbound[np.argmax(np.abs(errors[j] - errors[nonbound]))]
                    if take(first, j):
                        changed += 1
                        continue
                for candidates in (nonbound, range(n)):
                    if any(take(i, j) for i in cyclic(candidates)):
                        changed += 1
                        break
            iterations = iteration
            if examine_all and changed == 0:
                if self._kkt_residual(K) <= self.tol:
                    return iterations, True
                passes += 1
                if passes >= self.max_passes:
                    break
            elif changed:
                passes = 0
            if examine_all:
                examine_all = False
            elif changed == 0:
                examine_all = True
        return iterations, self._kkt_residual(K) <= self.tol

    def _kkt_residual(self, K):
        a, y = self._alphas, self._y_train
        bound_eps = min(1e-10, float(self.C) * 1e-8)
        violation = y * (K @ (a * y) + self._b) - 1
        lower = np.where(a < self.C - bound_eps, -violation, 0)
        upper = np.where(a > bound_eps, violation, 0)
        return float(max(np.max(lower), np.max(upper)))

    def _bias(self, K):
        # Free support vectors set b; otherwise average the KKT-allowed bias interval.
        y, a, C = self._y_train, self._alphas, float(self.C)
        raw = (a * y) @ K
        bound_eps = min(1e-10, C * 1e-8)
        free = (a > bound_eps) & (a < C - bound_eps)
        if free.any():
            return float(np.mean(y[free] - raw[free]))
        lower, upper = [], []
        for yi, ai, value in zip(y, a, raw):
            bound = yi - value
            if (yi > 0 and ai <= bound_eps) or (yi < 0 and ai >= C - bound_eps):
                lower.append(bound)
            if (yi < 0 and ai <= bound_eps) or (yi > 0 and ai >= C - bound_eps):
                upper.append(bound)
        return (
            float((max(lower) + min(upper)) / 2)
            if lower and upper
            else float(np.mean(y - raw))
        )

    def _check_is_fitted(self):
        if not hasattr(self, "fit_status_"):
            raise RuntimeError("SVM has not been fitted")

    def decision_function(self, X):
        self._check_is_fitted()
        X = self._features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; expected {self.n_features_in_}"
            )
        if self.kernel == "linear":
            scores = X @ self.coef_[0] + self.intercept_[0]
        elif self.kernel in ("poly", "polynomial"):
            K = polynomial_kernel(
                X,
                self.support_vectors_,
                degree=self.degree,
                gamma=self._gamma,
                coef0=self.coef0,
            )
            scores = K @ self.dual_coef_[0] + self.intercept_[0]
        else:
            K = gaussian_kernel(X, self.support_vectors_, gamma=self._gamma)
            scores = K @ self.dual_coef_[0] + self.intercept_[0]
        if not np.isfinite(scores).all():
            raise FloatingPointError("decision scores are non-finite")
        return scores

    def predict(self, X):
        self._check_is_fitted()
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]

    def score(self, X, y):
        predictions = self.predict(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(predictions):
            raise ValueError("y must be one-dimensional and match X samples")
        return float(np.mean(predictions == y))

    def save(self, path):
        """Pickle the fitted estimator; only load files from trusted sources."""
        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise TypeError("Saved object is not an SVM")
        model._check_is_fitted()
        return model
