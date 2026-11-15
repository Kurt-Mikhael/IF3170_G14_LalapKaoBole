import pickle
import warnings
from numbers import Integral, Real
from pathlib import Path

import numpy as np


class LogisticRegressionClassifier:
    def __init__(
        self,
        learning_rate=0.1,
        max_iter=2000,
        penalty=None,
        reg_strength=0.01,
        l1_ratio=0.5,
        tol=1e-7,
        patience=10,
        batch_size=None,
        random_state=42,
        record_history=False,
    ):
        self.learning_rate, self.max_iter = learning_rate, max_iter
        self.penalty, self.reg_strength, self.l1_ratio = penalty, reg_strength, l1_ratio
        self.tol, self.patience = tol, patience
        self.batch_size, self.random_state = batch_size, random_state
        self.record_history = record_history

    def get_params(self, deep=True):
        return {
            name: getattr(self, name)
            for name in (
                "learning_rate", "max_iter", "penalty", "reg_strength", "l1_ratio",
                "tol", "patience", "batch_size", "random_state", "record_history",
            )
        }

    def set_params(self, **params):
        unknown = set(params) - set(self.get_params())
        if unknown:
            raise ValueError(f"Invalid parameter(s): {', '.join(sorted(unknown))}")
        for name, value in params.items():
            setattr(self, name, value)
        self._clear_fitted()
        return self

    def _clear_fitted(self):
        for name in (
            "classes_", "coef_", "intercept_", "n_features_in_", "n_iter_",
            "converged_", "fit_status_", "loss_history_", "parameter_history_",
        ):
            self.__dict__.pop(name, None)

    def _validate_params(self):
        for name in ("learning_rate", "reg_strength", "l1_ratio", "tol"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real number")
        if self.learning_rate <= 0 or self.reg_strength < 0 or self.tol < 0:
            raise ValueError("learning_rate must be positive; strength and tol nonnegative")
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be between zero and one")
        if self.penalty not in (None, "l1", "l2", "elasticnet"):
            raise ValueError("penalty must be None, l1, l2, or elasticnet")
        for name in ("max_iter", "patience", "batch_size"):
            value = getattr(self, name)
            if name == "batch_size" and value is None:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.random_state is not None and (
            isinstance(self.random_state, (bool, np.bool_))
            or not isinstance(self.random_state, Integral) or self.random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer or None")
        if not isinstance(self.record_history, (bool, np.bool_)):
            raise ValueError("record_history must be boolean")

    @staticmethod
    def _features(X):
        if np.iscomplexobj(X):
            raise ValueError("X must be real")
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must be a finite numeric 2D array") from exc
        if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite 2D feature matrix")
        return X

    @staticmethod
    def _sigmoid(z):
        # Hindari overflow untuk skor dengan nilai mutlak besar.
        e = np.exp(-np.abs(z))
        return np.where(z >= 0, 1 / (1 + e), e / (1 + e))

    def _regularization(self):
        if self.penalty is None:
            return 0.0, 0.0
        ratio = {"l1": 1.0, "l2": 0.0, "elasticnet": self.l1_ratio}[self.penalty]
        return self.reg_strength * ratio, self.reg_strength * (1 - ratio)

    def _loss_gradient(self, X, y, w, b):
        l1, l2 = self._regularization()
        z = X @ w + b
        loss = np.mean(np.logaddexp(0, (1 - 2 * y) * z))
        loss += l1 * np.abs(w).sum() + 0.5 * l2 * (w @ w)
        residual = self._sigmoid(z) - y
        # Penalti hanya untuk bobot; L1 memakai subgradien.
        dw = X.T @ residual / len(y) + l1 * np.sign(w) + l2 * w
        db = residual.mean()
        return float(loss), dw, float(db)

    def fit(self, X, y):
        self._clear_fitted()
        self._validate_params()
        X = self._features(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional with one label per sample")
        if y.dtype.kind not in "biufUS" or (y.dtype.kind in "f" and not np.isfinite(y).all()):
            raise ValueError("y must contain finite numeric or string labels")
        classes, encoded = np.unique(y, return_inverse=True)
        if len(classes) != 2:
            raise ValueError("binary classification requires exactly two classes")
        encoded = encoded.astype(float)
        rng = np.random.default_rng(self.random_state)
        try:
            w, b, iterations, converged = self._fit_gradient_descent(X, encoded, rng)
        except Exception:
            self._clear_fitted()
            raise
        self.coef_ = w[None, :]
        self.intercept_ = np.array([b])
        self.n_features_in_ = X.shape[1]
        self.classes_ = classes
        self.n_iter_, self.converged_, self.fit_status_ = (
            iterations, bool(converged), 0 if converged else 1,
        )
        if not converged:
            warnings.warn("Logistic regression stopped before convergence", RuntimeWarning)
        return self

    def _fit_gradient_descent(self, X, y, rng):
        w, b = np.zeros(X.shape[1]), 0.0
        self.loss_history_ = [self._loss_gradient(X, y, w, b)[0]]
        self.parameter_history_ = [(w.copy(), b)] if self.record_history else []
        stable_epochs = 0
        size = len(X) if self.batch_size is None else min(self.batch_size, len(X))
        for epoch in range(1, self.max_iter + 1):
            indices = rng.permutation(len(X)) if size < len(X) else None
            for start in range(0, len(X), size):
                batch = slice(None) if indices is None else indices[start:start + size]
                with np.errstate(over="ignore", invalid="ignore"):
                    _, dw, db = self._loss_gradient(X[batch], y[batch], w, b)
                    w -= self.learning_rate * dw
                    b -= self.learning_rate * db
                if not np.isfinite(w).all() or not np.isfinite(b):
                    raise FloatingPointError("Nonfinite parameters; scale features or reduce learning_rate")
            with np.errstate(over="ignore", invalid="ignore"):
                loss = self._loss_gradient(X, y, w, b)[0]
            if not np.isfinite(loss):
                raise FloatingPointError("Nonfinite objective; scale features or reduce learning_rate")
            previous_loss = self.loss_history_[-1]
            self.loss_history_.append(loss)
            if self.record_history:
                self.parameter_history_.append((w.copy(), float(b)))
            if abs(previous_loss - loss) <= self.tol * max(1.0, abs(previous_loss)):
                stable_epochs += 1
            else:
                stable_epochs = 0
            if stable_epochs >= self.patience:
                return w, b, epoch, True
        return w, b, epoch, False

    def _check_is_fitted(self):
        if not hasattr(self, "fit_status_"):
            raise RuntimeError("LogisticRegressionClassifier has not been fitted")

    def decision_function(self, X):
        self._check_is_fitted()
        X = self._features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features; expected {self.n_features_in_}")
        with np.errstate(over="ignore", invalid="ignore"):
            scores = X @ self.coef_[0] + self.intercept_[0]
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite decision scores")
        return scores

    def predict_proba(self, X):
        positive = self._sigmoid(self.decision_function(X))
        return np.column_stack((1 - positive, positive))

    def predict(self, X, threshold=0.5):
        if (
            isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, Real)
            or not np.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("threshold must be a finite number between zero and one")
        probabilities = self.predict_proba(X)
        return self.classes_[(probabilities[:, 1] >= threshold).astype(int)]

    def score(self, X, y):
        predictions = self.predict(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(predictions):
            raise ValueError("y must be one-dimensional and match X samples")
        return float(np.mean(predictions == y))

    def save(self, path):
        self._check_is_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise TypeError("The saved object is not a LogisticRegressionClassifier")
        model._check_is_fitted()
        return model
