import math
import pickle
from pathlib import Path
import numpy as np

# struktur bintang: kelas -> tiap fitur
class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        # alpha: hitungan palsu yang ditambahkan ke tiap nilai fitur dan ke tiap kelas (smoothing)
        if alpha < 0:
            raise ValueError("alpha must be non-negative")

        self.alpha = alpha
        self.classes_ = np.array([])
        self.class_log_prior_ = []
        self.feature_params_ = []

    def _check_X(self, X):
        # NaN boleh (missing), inf tidak
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.size == 0:
            raise ValueError("X must be a non-empty 2D array")
        if np.isinf(X).any():
            raise ValueError("X must not contain infinite values")
        return X

    def _check_is_fitted(self):
        if len(self.classes_) == 0:
            raise RuntimeError("NaiveBayesClassifier has not been fitted")

    def _log(self, x):
        # log(0) tidak terdefinisi, dianggap -tak hingga
        if x == 0:
            return -math.inf
        return math.log(x)

    def fit(self, X, y):
        # fit = menghitung frekuensi
        X = self._check_X(X)
        y = np.asarray(y).reshape(-1)
        if len(y) != len(X):
            raise ValueError("X and y must have the same number of samples")

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # P(kelas) = (hitungan kelas + alpha) / (jumlah baris + alpha * jumlah kelas)
        self.class_log_prior_ = []
        for c in self.classes_:
            n_in_class = np.sum(y == c)
            self.class_log_prior_.append(math.log((n_in_class + self.alpha) / (n_samples + self.alpha * n_classes)))

        # semua fitur diperlakukan sebagai kategorikal
        self.feature_params_ = []
        for j in range(n_features):
            self.feature_params_.append(self._fit_categorical(X[:, j], y))
        return self

    def _fit_categorical(self, column, y):
        # nilai yang pernah muncul di fitur ini (NaN diabaikan)
        not_missing = ~np.isnan(column)
        values = sorted(set(column[not_missing]))
        n_values = len(values)

        log_prob = [] # log_prob[k][nilai] = log P(nilai | kelas ke-k)
        log_unseen = [] # nilai yang tidak pernah dilihat di kelas ke-k
        for c in self.classes_:
            in_class = column[(y == c) & not_missing]
            # (hitungan + alpha) / (total kelas + alpha * K)
            denominator = len(in_class) + self.alpha * n_values

            table = {}
            for v in values:
                count = np.sum(in_class == v)
                if denominator == 0:
                    table[v] = 0.0  # kelas ini tidak punya data: netral
                else:
                    table[v] = self._log(count + self.alpha) - math.log(denominator)
            log_prob.append(table)

            if denominator == 0:
                log_unseen.append(0.0)
            else:
                log_unseen.append(self._log(self.alpha) - math.log(denominator))

        return {"log_prob": log_prob, "log_unseen": log_unseen}

    def _log_likelihood(self, param, k, value):
        # log P(fitur = value | kelas ke-k)
        table = param["log_prob"][k]
        if value in table:
            return table[value]
        return param["log_unseen"][k]

    def _log_scores(self, X):
        # skor = log P(kelas) + jumlah log P(fitur | kelas)
        self._check_is_fitted()
        X = self._check_X(X)
        if X.shape[1] != len(self.feature_params_):
            raise ValueError("X has a different number of features than the training data")

        all_scores = []
        for row in X:
            scores = list(self.class_log_prior_)
            for j in range(len(row)):
                value = row[j]
                if np.isnan(value):
                    continue # NaN dilewati
                for k in range(len(self.classes_)):
                    scores[k] += self._log_likelihood(self.feature_params_[j], k, value)
            all_scores.append(scores)
        return all_scores

    def predict_proba(self, X):
        # normalisasi skor -> P(kelas | fitur); kolom mengikuti classes_
        probabilities = []
        for scores in self._log_scores(X):
            best = max(scores)
            if best == -math.inf:
                # semua kelas bernilai 0 (hanya mungkin saat alpha=0): pakai prior
                probabilities.append([math.exp(p) for p in self.class_log_prior_])
                continue
            # kurangi skor terbesar agar exp() tidak underflow (softmax yang stabil, Blanchard dkk., 2021)
            exps = [math.exp(s - best) for s in scores]
            total = sum(exps)
            probabilities.append([e / total for e in exps])
        return np.array(probabilities)

    def predict(self, X):
        # argmax dari predict_proba
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def save(self, path):
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
            raise TypeError("The saved object is not a NaiveBayesClassifier")
        return model