
import pickle
import numpy as np

EPS = 1e-12


def entropy(counts):
    # counts bisa 1D, atau 2D (entropy dihitung per baris)
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    log_p = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -(p * log_p).sum(axis=-1)


def split_information(sizes):
    # SplitInfo(S, A) = -sum |Si|/|S| log2 |Si|/|S|  (slide 26)
    return entropy(np.asarray(sizes, dtype=float))


def is_missing(value):
    return value is None or value != value 


class TreeNode:
    def __init__(self, counts):
        self.distribution = counts / counts.sum()  # proporsi bobot tiap kelas di node ini
        self.prediction = int(np.argmax(counts))   # indeks kelas mayoritas di node ini
        self.feature = None           # atribut yang diuji, None berarti leaf
        self.threshold = None         # None berarti atribut kategorikal
        self.children = {}            # nilai kategori / "left" / "right" -> TreeNode
        self.probabilities = {}       # cabang -> p_i, proporsi bobot data training (nilai diketahui) di cabang itu
        self.validation_errors = 0.0  # error validation (berbobot) jika node ini jadi leaf, dipakai saat prune

    def is_leaf(self):
        return self.feature is None


class C45DecisionTree:

    def fit(self, X, y):
        self.feature_names = list(X.columns)
        self.classes, self.train_y = np.unique(np.asarray(y), return_inverse=True)

        self.is_categorical, self.train_columns, self.train_missing = [], [], []
        for name in self.feature_names:
            if X[name].dtype == object:
                column = X[name].to_numpy(dtype=object)
                missing = np.array([is_missing(v) for v in column])
            else:
                column = X[name].to_numpy(dtype=float)
                missing = np.isnan(column)
            self.is_categorical.append(X[name].dtype == object)
            self.train_columns.append(column)
            self.train_missing.append(missing)

        # setiap data training mulai dengan bobot 1
        all_rows = np.arange(len(self.train_y))
        self.root = self.build_tree(all_rows, np.ones(len(all_rows)), set(range(len(self.feature_names))))

        del self.train_columns, self.train_missing, self.train_y  # data training tidak ikut disimpan
        return self

    def build_tree(self, rows, weights, attributes):
        # bobot tiap data, bisa pecahan untuk data yang nilainya missing di atasnya
        counts = np.bincount(self.train_y[rows], weights=weights, minlength=len(self.classes))
        node = TreeNode(counts)

        if np.count_nonzero(counts) <= 1 or not attributes:
            return node

        best = self.find_best_split(rows, weights, attributes)
        if best is None:
            return node
        feature, threshold = best

        column = self.train_columns[feature][rows]
        missing = self.train_missing[feature][rows]
        if threshold is None:
            branches = {v: ~missing & (column == v) for v in np.unique(column[~missing])}
        else:
            branches = {"left": ~missing & (column <= threshold),
                        "right": ~missing & (column > threshold)}

        node.feature = feature
        node.threshold = threshold

        child_attributes = attributes - {feature} if threshold is None else attributes
        known_weight = weights[~missing].sum()
        for key, mask in branches.items():
            # Missing value data ikut ke semua cabang dengan bobot dikali p_i
            p = weights[mask].sum() / known_weight
            node.probabilities[key] = p
            child_rows = np.concatenate([rows[mask], rows[missing]])
            child_weights = np.concatenate([weights[mask], weights[missing] * p])
            node.children[key] = self.build_tree(child_rows, child_weights, child_attributes)
        return node

    def find_best_split(self, rows, weights, attributes):
        # Pilih atribut dengan Gain Ratio terbesar
        best, best_ratio = None, 0.0
        for feature in sorted(attributes):
            result = self.evaluate_attribute(rows, weights, feature)
            if result is not None and result[0] > best_ratio:
                best_ratio, best = result[0], (feature, result[1])
        return best

    def evaluate_attribute(self, rows, weights, feature):
        # Mengembalikan (gain_ratio, threshold), atau None jika atribut tidak bisa dipakai
        known = ~self.train_missing[feature][rows]
        if not known.any():
            return None
        values = self.train_columns[feature][rows][known]
        y = self.train_y[rows][known]
        w = weights[known]
        fraction_known = w.sum() / weights.sum()
        n_classes = len(self.classes)

        if self.is_categorical[feature]:
            threshold = None
            categories, category_index = np.unique(values, return_inverse=True)
            counts = np.zeros((len(categories), n_classes))
            # hitung jumlah label per kategori , hasilnya berupa matriks 2 dimensi
            """
            contoh:
            values = ['A', 'B', 'A', 'C', 'B']
            y = [0, 1, 0, 1, 0]
            categories = ['A', 'B', 'C']
            category_index = [0, 1, 0, 2, 1]
            counts = [[2, 0],  # kategori A: 2 label 0
                      [1, 1],  # kategori B: 1 label 0, 1 label 1
                      [0, 1]]  # kategori C: 0 label 0, 1 label 1
            """
            np.add.at(counts, (category_index, y), w)
            # jika hanya ada 1 kategori, data tidak terbagi sehingga atribut tidak bisa dipakai
            if len(categories) < 2:
                return None
        else:
            # urutin nilai, kandidat threshold = titik tengah nilai bersebelahan 
            order = np.argsort(values, kind="mergesort")
            values, y, w = values[order], y[order], w[order]
            cumulative = np.cumsum(np.eye(n_classes)[y] * w[:, None], axis=0)
            cuts = np.nonzero(values[1:] > values[:-1])[0]
            if len(cuts) == 0:
                return None

            left = cumulative[cuts]
            right = cumulative[-1] - left
            n_left, n_right = left.sum(axis=1), right.sum(axis=1)
            gains = entropy(cumulative[-1]) - (n_left * entropy(left) + n_right * entropy(right)) / w.sum()
            best = int(np.argmax(gains))

            # Threshold dipilih dengan Gain terbesar 
            lower, upper = values[cuts[best]], values[cuts[best] + 1]
            threshold = (lower + upper) / 2
            if not lower <= threshold < upper:
                threshold = lower
            counts = np.vstack([left[best], right[best]])

        # Gain hanya dari data yang nilainya diketahui, dikali fraksinya 
        sizes = counts.sum(axis=1)
        gain = fraction_known * (entropy(counts.sum(axis=0)) - (sizes / sizes.sum() * entropy(counts)).sum())
        split_info = split_information(sizes)
        if gain <= EPS or split_info <= EPS:
            return None
        return gain / split_info, threshold

    def predict(self, X):
        rows = X[self.feature_names].to_numpy(dtype=object)
        return self.classes[[int(np.argmax(self.predict_distribution(self.root, row))) for row in rows]]

    def predict_distribution(self, node, row):
        # mengembalikan proporsi kelas untuk satu baris data
        if node.is_leaf():
            return node.distribution
        key = self.branch_key(node, row)
        if key is None:
            # nilai missing / kategori tidak dikenal: hasil semua cabang digabung dengan bobot p_i
            return sum(p * self.predict_distribution(node.children[k], row) for k, p in node.probabilities.items())
        return self.predict_distribution(node.children[key], row)

    def branch_key(self, node, row):
        # menentukan cabang yang diikuti satu baris data
        value = row[node.feature]
        if is_missing(value):
            return None
        if node.threshold is None:
            return value if value in node.children else None
        return "left" if value <= node.threshold else "right"

    def prune(self, X_val, y_val):
        # Reduced Error Pruning diulangi sampai pemangkasan berikutnya merugikan
        rows = X_val[self.feature_names].to_numpy(dtype=object)
        y = np.searchsorted(self.classes, np.asarray(y_val))
        self.count_validation_errors(self.root, rows, y, np.ones(len(y)))

        while True:
            candidates = []
            self.collect_candidates(self.root, candidates)
            if not candidates:
                break
            # pilih node yang paling mengurangi error validation
            gain, size, node = max(candidates, key=lambda candidate: candidate[:2])
            # hanya pangkas jika akurasi validation naik
            if gain <= 1e-9:  # toleransi pembulatan karena error berupa bobot pecahan
                break
            node.feature, node.threshold = None, None
            node.children, node.probabilities = {}, {}
        return self

    def count_validation_errors(self, node, rows, y, weights):
        # jumlah bobot salah prediksi pada data validation jika node ini dijadikan leaf
        node.validation_errors = float(np.sum(weights * (y != node.prediction)))
        if node.is_leaf():
            return
        keys = np.array([self.branch_key(node, row) for row in rows], dtype=object)
        unknown = np.array([key is None for key in keys], dtype=bool)
        for key, child in node.children.items():
            # data validation yang nilainya missing juga ikut ke semua cabang dengan bobot p_i
            mask = keys == key
            self.count_validation_errors(child,
                                         np.concatenate([rows[mask], rows[unknown]]),
                                         np.concatenate([y[mask], y[unknown]]),
                                         np.concatenate([weights[mask], weights[unknown] * node.probabilities[key]]))

    def collect_candidates(self, node, candidates):
        # mengembalikan (error subtree, jumlah node subtree), dan mencatat (gain, size, node) tiap node internal
        if node.is_leaf():
            return node.validation_errors, 1
        subtree_errors, size = 0, 1
        for child in node.children.values():
            child_errors, child_size = self.collect_candidates(child, candidates)
            subtree_errors += child_errors
            size += child_size
        candidates.append((subtree_errors - node.validation_errors, size, node))
        return subtree_errors, size

    def count_nodes(self, node=None):
        node = self.root if node is None else node
        return 1 + sum(self.count_nodes(child) for child in node.children.values())

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
