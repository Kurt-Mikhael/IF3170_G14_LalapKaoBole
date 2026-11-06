from __future__ import annotations
from pathlib import Path
import pickle
from typing import Any, Literal
import numpy as np

class KNNClassifier:
    def __init__(self, n_neighbors: int = 5, weights: Literal["uniform", "distance"] = "uniform") -> None:
        if not isinstance(n_neighbors, (int, np.integer)) or isinstance(n_neighbors, bool):
            raise TypeError("n_neighbors must be a positive integer")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be either 'uniform' or 'distance'")

        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self._X_train: np.ndarray | None = None
        self._y_encoded: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    @staticmethod
    def _as_2d_features(X: Any) -> np.ndarray:
        # mengubah input menjadi matriks fitur 2D bertipe float yang valid
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        if X_array.ndim != 2:
            raise ValueError("X must be a two-dimensional array-like object")
        if X_array.shape[0] == 0 or X_array.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature")
        if not np.isfinite(X_array).all():
            raise ValueError("X must not contain NaN or infinite values")
        return X_array

    def fit(self, X: Any, y: Any) -> "KNNClassifier":
        # menyimpan data latih dan label untuk pencarian tetangga terdekat
        X_array = self._as_2d_features(X)
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            y_array = y_array.reshape(-1)
        if len(y_array) != len(X_array):
            raise ValueError("X and y must have the same number of samples")
        if len(y_array) == 0:
            raise ValueError("y must contain at least one label")
        if self.n_neighbors > len(X_array):
            raise ValueError("n_neighbors cannot exceed the number of training samples")

        classes, y_encoded = np.unique(y_array, return_inverse=True)
        self._X_train = X_array.copy()
        self._y_encoded = y_encoded
        self.classes_ = classes

        return self

    def _check_is_fitted(self) -> None:
        if self._X_train is None or self._y_encoded is None or self.classes_ is None:
            raise RuntimeError("KNNClassifier has not been fitted")

    def _neighbor_indices_and_distances(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X_array = self._as_2d_features(X)
        assert self._X_train is not None
        if X_array.shape[1] != self._X_train.shape[1]:
            raise ValueError(
                f"X has {X_array.shape[1]} features, but the model expects "
                f"{self._X_train.shape[1]}"
            )

        # hitung jarak euclidean
        squared_distances = np.sum((X_array[:, None, :] - self._X_train[None, :, :]) ** 2, axis=2)
        distances = np.sqrt(np.maximum(squared_distances, 0.0))
        nearest = np.argsort(distances, axis=1, kind="stable")[:, : self.n_neighbors]

        return nearest, np.take_along_axis(distances, nearest, axis=1)

    def predict_proba(self, X: Any) -> np.ndarray:
        # menghitung probabilitas tiap kelas berdasarkan nearest neighbor
        self._check_is_fitted()
        nearest, distances = self._neighbor_indices_and_distances(X)
        assert self._y_encoded is not None and self.classes_ is not None

        neighbour_labels = self._y_encoded[nearest]
        probabilities = np.zeros((len(nearest), len(self.classes_)), dtype=float)
        for row_index, (labels, row_distances) in enumerate(zip(neighbour_labels, distances)):
            if self.weights == "uniform":
                votes = np.ones_like(row_distances)
            else:
                # jika jarak 0, gunakan sampel yang sama agar tidak terjadi pembagian dengan nol
                exact_matches = row_distances == 0
                votes = exact_matches.astype(float) if exact_matches.any() else 1.0 / row_distances
            probabilities[row_index] = np.bincount(
                labels, weights=votes, minlength=len(self.classes_)
            )
            probabilities[row_index] /= probabilities[row_index].sum()
            
        return probabilities

    def predict(self, X: Any) -> np.ndarray:
        # memprediksi kelas dengan suara tertinggi untuk setiap sampel
        self._check_is_fitted()
        assert self.classes_ is not None
        # pilih kelas pertama pada classes_ jika hasil voting seri
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> "KNNClassifier":
        with Path(path).open("rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise TypeError("The saved object is not a KNNClassifier")
        model._check_is_fitted()
        return model