import math
from collections import Counter

import numpy as np

# Mutual information (Cover & Thomas, 2006) dan conditional mutual information (Friedman dkk., 1997) untuk graf dan pohon TAN
# Koreksi bias Miller-Madow (Miller, 1955) bersifat opsional

def _clean_pairs(columns, y):
    # buang baris yang NaN di salah satu kolom, sisanya jadi list nilai
    keep = ~np.isnan(y.astype(float))
    for column in columns:
        keep = keep & ~np.isnan(column)
    return [column[keep].tolist() for column in columns], y[keep].tolist()


def mutual_information(column, y, bias_correction=True):
    # I(fitur; kelas) dalam bit, dengan opsi koreksi Miller-Madow agar fitur bernilai banyak tidak tampak lebih berkaitan
    (values,), labels = _clean_pairs([column], y)
    n = len(values)
    if n == 0:
        return 0.0
    n_value = Counter(values)
    n_label = Counter(labels)
    n_both = Counter(zip(values, labels))

    total = 0.0
    for (v, c), count in n_both.items():
        total += count / n * math.log2(count * n / (n_value[v] * n_label[c]))

    if not bias_correction:
        return max(total, 0.0)
    bias = (len(n_value) - 1) * (len(n_label) - 1) / (2 * n * math.log(2))
    return max(total - bias, 0.0)


def conditional_mutual_information(column_a, column_b, y, bias_correction=True):
    # I(A; B | kelas) dalam bit: seberapa bergantung dua fitur setelah kelas diketahui
    (a_values, b_values), labels = _clean_pairs([column_a, column_b], y)
    n = len(labels)
    if n == 0:
        return 0.0
    n_label = Counter(labels)
    n_a_label = Counter(zip(a_values, labels))
    n_b_label = Counter(zip(b_values, labels))
    n_all = Counter(zip(a_values, b_values, labels))

    total = 0.0
    for (a, b, c), count in n_all.items():
        total += count / n * math.log2(count * n_label[c] / (n_a_label[(a, c)] * n_b_label[(b, c)]))

    if not bias_correction:
        return max(total, 0.0)
    bias = (len(set(a_values)) - 1) * (len(set(b_values)) - 1) * len(n_label) / (2 * n * math.log(2))
    return max(total - bias, 0.0)


def feature_relations(X, y, feature_names):
    # skor tiap fitur terhadap kelas, dan skor tiap pasangan fitur (diurutkan dari yang terbesar)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError("feature_names must have one name per column of X")
    if len(y) != len(X):
        raise ValueError("X and y must have the same number of samples")

    class_scores = {}
    for j, name in enumerate(feature_names):
        class_scores[name] = mutual_information(X[:, j], y)

    pair_scores = []
    for a in range(X.shape[1]):
        for b in range(a + 1, X.shape[1]):
            score = conditional_mutual_information(X[:, a], X[:, b], y)
            pair_scores.append((feature_names[a], feature_names[b], score))
    pair_scores.sort(key=lambda item: item[2], reverse=True)
    return class_scores, pair_scores