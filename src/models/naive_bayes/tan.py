import math
import pickle
from pathlib import Path
import numpy as np

from .information import conditional_mutual_information, mutual_information

# Tree Augmented Naive Bayes: tiap fitur punya satu induk fitur tambahan sehingga fitur membentuk pohon (Friedman dkk., 1997)
# Pohonnya dipelajari dengan Construct-TAN, yang memakai pohon ketergantungan Chow & Liu (1968) dan pohon rentang Kruskal (1956)
class TreeAugmentedNaiveBayes:
    def __init__(self, alpha=1.0, smoothing="laplace", n0=5.0, root=None, bias_correction=False):
        # alpha dipakai untuk prior kelas dan smoothing Laplace, sedangkan n0 dipakai untuk smoothing Dirichlet
        # root adalah indeks fitur akar, bawaannya fitur dengan informasi kelas terbesar
        # bias_correction memakai koreksi Miller-Madow pada bobot pohon, sedangkan makalah memakai nilai mentah
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if smoothing not in ("laplace", "dirichlet"):
            raise ValueError("smoothing must be 'laplace' or 'dirichlet'")
        if n0 <= 0:
            raise ValueError("n0 must be positive")

        self.alpha = alpha
        self.smoothing = smoothing
        self.n0 = n0
        self.root = root
        self.bias_correction = bias_correction
        self.classes_ = np.array([])
        self.class_log_prior_ = []
        self.parents_ = [] # indeks fitur induk tiap fitur, None untuk akar
        self.edges_ = [] # (induk, anak, bobot) pohon yang terbentuk
        self.counts_ = [] # hitungan per (kelas, nilai induk, nilai)
        self.totals_ = [] # jumlah baris per (kelas, nilai induk)
        self.marginals_ = []
        self.n_values_ = []

    def _check_X(self, X):
        # nilai kosong harus sudah dikodekan menjadi kategori sebelum masuk ke model ini
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.size == 0:
            raise ValueError("X must be a non-empty 2D array")
        if not np.isfinite(X).all():
            raise ValueError("X must not contain NaN or infinite values; encode missing values as a category first")
        return X

    def _check_is_fitted(self):
        if len(self.classes_) == 0:
            raise RuntimeError("TreeAugmentedNaiveBayes has not been fitted")

    def _log(self, x):
        # log(0) tidak terdefinisi, dianggap -tak hingga
        if x == 0:
            return -math.inf
        return math.log(x)

    def fit(self, X, y):
        X = self._check_X(X)
        y = np.asarray(y).reshape(-1)
        if len(y) != len(X):
            raise ValueError("X and y must have the same number of samples")

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_index = np.searchsorted(self.classes_, y)

        # P(kelas) = (hitungan kelas + alpha) / (jumlah baris + alpha * jumlah kelas)
        self.class_log_prior_ = []
        for k in range(n_classes):
            n_in_class = np.sum(class_index == k)
            self.class_log_prior_.append(math.log((n_in_class + self.alpha) / (n_samples + self.alpha * n_classes)))

        self._learn_tree(X, class_index)

        # hitungan tiap fitur dikondisikan pada kelas dan induk fiturnya
        self.counts_ = []
        self.totals_ = []
        self.marginals_ = []
        self.n_values_ = []
        for j in range(n_features):
            parent = self.parents_[j]
            counts = {}
            totals = {}
            marginal = {}
            for i in range(n_samples):
                value = X[i, j]
                parent_value = None if parent is None else X[i, parent]
                k = int(class_index[i])
                counts[(k, parent_value, value)] = counts.get((k, parent_value, value), 0) + 1
                totals[(k, parent_value)] = totals.get((k, parent_value), 0) + 1
                marginal[value] = marginal.get(value, 0) + 1
            self.counts_.append(counts)
            self.totals_.append(totals)
            self.marginals_.append({value: count / n_samples for value, count in marginal.items()})
            self.n_values_.append(len(marginal))
        return self

    def _learn_tree(self, X, y):
        n_features = X.shape[1]

        # bobot sisi adalah conditional mutual information antara dua fitur
        weighted_edges = []
        for a in range(n_features):
            for b in range(a + 1, n_features):
                weight = conditional_mutual_information(X[:, a], X[:, b], y, bias_correction=self.bias_correction)
                weighted_edges.append((weight, a, b))

        # pohon rentang maksimum dengan algoritma Kruskal, dan bobot yang sama diurutkan menurut indeks agar hasilnya tetap
        weighted_edges.sort(key=lambda edge: (-edge[0], edge[1], edge[2]))
        group = list(range(n_features))

        def find(x):
            while group[x] != x:
                group[x] = group[group[x]]
                x = group[x]
            return x

        tree_edges = []
        for weight, a, b in weighted_edges:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                group[root_a] = root_b
                tree_edges.append((a, b, weight))

        # akar dipilih lalu semua sisi diarahkan menjauhi akar
        if self.root is None:
            class_scores = [mutual_information(X[:, j], y, bias_correction=self.bias_correction) for j in range(n_features)]
            root = int(np.argmax(class_scores))
        else:
            root = int(self.root)
            if root < 0 or root >= n_features:
                raise ValueError("root must be a valid feature index")

        neighbors = {j: [] for j in range(n_features)}
        for a, b, weight in tree_edges:
            neighbors[a].append((b, weight))
            neighbors[b].append((a, weight))

        self.parents_ = [None] * n_features
        self.edges_ = []
        visited = {root}
        queue = [root]
        while queue:
            current = queue.pop(0)
            for other, weight in neighbors[current]:
                if other not in visited:
                    visited.add(other)
                    self.parents_[other] = current
                    self.edges_.append((current, other, weight))
                    queue.append(other)

    def _log_cond(self, j, k, parent_value, value):
        # log P(fitur j = value | kelas ke-k, induk fitur = parent_value)
        count = self.counts_[j].get((k, parent_value, value), 0)
        total = self.totals_[j].get((k, parent_value), 0)
        if self.smoothing == "laplace":
            denominator = total + self.alpha * self.n_values_[j]
            if denominator == 0:
                return 0.0
            return self._log(count + self.alpha) - math.log(denominator)
        # Dirichlet memperhalus ke frekuensi nilai di seluruh data dengan bobot n0 (Friedman dkk., 1997)
        prior = self.marginals_[j].get(value, 0.0)
        return self._log(count + self.n0 * prior) - math.log(total + self.n0)

    def _log_scores(self, X):
        # skor = log P(kelas) + jumlah log P(fitur | kelas, induk fitur)
        self._check_is_fitted()
        X = self._check_X(X)
        if X.shape[1] != len(self.parents_):
            raise ValueError("X has a different number of features than the training data")

        all_scores = []
        for row in X:
            scores = list(self.class_log_prior_)
            for k in range(len(self.classes_)):
                for j in range(len(row)):
                    parent = self.parents_[j]
                    parent_value = None if parent is None else row[parent]
                    scores[k] += self._log_cond(j, k, parent_value, row[j])
            all_scores.append(scores)
        return all_scores

    def predict_proba(self, X):
        # normalisasi skor -> P(kelas | fitur), kolom mengikuti classes_
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
            raise TypeError("The saved object is not a TreeAugmentedNaiveBayes")
        return model