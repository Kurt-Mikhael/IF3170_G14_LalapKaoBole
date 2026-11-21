import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def _clean_pairs(columns, y):
    # buang baris yang NaN di salah satu kolom, sisanya jadi list nilai
    keep = ~np.isnan(y.astype(float))
    for column in columns:
        keep = keep & ~np.isnan(column)
    return [column[keep].tolist() for column in columns], y[keep].tolist()


def mutual_information(column, y):
    # I(fitur; kelas) dalam bit, dikurangi koreksi bias supaya fitur bernilai banyak tidak unggul
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

    bias = (len(n_value) - 1) * (len(n_label) - 1) / (2 * n * math.log(2))
    return max(total - bias, 0.0)


def conditional_mutual_information(column_a, column_b, y):
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


def plot_naive_bayes_graph(X, y, feature_names, class_name="kelas", top_pairs=8, ax=None, title=None):
    # Garis abu-abu: kelas -> fitur (struktur Naive Bayes), tebal = I(fitur; kelas)
    # Garis merah putus-putus: pasangan fitur yang masih bergantung setelah kelas diketahui
    class_scores, pair_scores = feature_relations(X, y, feature_names)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    # fitur diletakkan melingkar, yang paling berkaitan dengan kelas dimulai dari atas
    ordered = sorted(feature_names, key=lambda name: class_scores[name], reverse=True)
    positions = {}
    for i, name in enumerate(ordered):
        angle = math.pi / 2 - 2 * math.pi * i / len(ordered)
        positions[name] = (math.cos(angle), math.sin(angle))

    strongest_class = max(class_scores.values())
    for name in ordered:
        x_pos, y_pos = positions[name]
        width = 0.5 + 5.0 * class_scores[name] / strongest_class if strongest_class > 0 else 0.5
        ax.annotate("", xy=(x_pos * 0.93, y_pos * 0.93), xytext=(0, 0),
                    arrowprops={"arrowstyle": "-|>", "color": "#8c8c8c", "lw": width, "shrinkA": 22, "shrinkB": 2})

    shown_pairs = [pair for pair in pair_scores[:top_pairs] if pair[2] > 0]
    strongest_pair = shown_pairs[0][2] if shown_pairs else 1.0
    for name_a, name_b, score in shown_pairs:
        (x1, y1), (x2, y2) = positions[name_a], positions[name_b]
        ax.plot([x1, x2], [y1, y2], linestyle="--", color="#c44e52", alpha=0.8,
                linewidth=0.8 + 2.5 * score / strongest_pair)

    ax.scatter([0], [0], s=2600, color="#dd8452", zorder=3)
    ax.text(0, 0, class_name, ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)
    for name in ordered:
        x_pos, y_pos = positions[name]
        ax.scatter([x_pos], [y_pos], s=260, color="#4c72b0", zorder=3)
        horizontal = "left" if x_pos > 0.1 else "right" if x_pos < -0.1 else "center"
        vertical = "bottom" if y_pos > 0.9 else "top" if y_pos < -0.9 else "center"
        ax.text(x_pos * 1.08, y_pos * 1.08, name, ha=horizontal, va=vertical, fontsize=9)

    ax.plot([], [], color="#8c8c8c", linewidth=3, label=f"{class_name} → fitur (tebal = I(fitur; kelas))")
    ax.plot([], [], color="#c44e52", linestyle="--", linewidth=2,
            label=f"{len(shown_pairs)} pasangan fitur paling bergantung (tebal = I(A; B | kelas))")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=8, frameon=False)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    return ax
