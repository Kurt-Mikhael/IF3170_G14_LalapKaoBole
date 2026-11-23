import math

import matplotlib.pyplot as plt

from .information import feature_relations

def plot_tan_structure(model, feature_names, class_name="kelas", ax=None, title=None):
    # garis abu-abu tipis: kelas -> fitur (seperti Naive Bayes)
    # panah merah: induk -> anak pada pohon antarfitur yang dipelajari TAN, tebal = I(A; B | kelas)
    if len(model.parents_) != len(feature_names):
        raise ValueError("feature_names must have one name per feature of the model")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    # fitur diletakkan melingkar menurut urutan yang diberikan
    positions = {}
    for i, name in enumerate(feature_names):
        angle = math.pi / 2 - 2 * math.pi * i / len(feature_names)
        positions[name] = (math.cos(angle), math.sin(angle))

    for name in feature_names:
        x_pos, y_pos = positions[name]
        ax.annotate("", xy=(x_pos * 0.93, y_pos * 0.93), xytext=(0, 0),
                    arrowprops={"arrowstyle": "-|>", "color": "#c8c8c8", "lw": 0.8, "shrinkA": 22, "shrinkB": 2})

    strongest = max([weight for _, _, weight in model.edges_], default=1.0)
    if strongest <= 0:
        strongest = 1.0
    for parent, child, weight in model.edges_:
        start = positions[feature_names[parent]]
        end = positions[feature_names[child]]
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "-|>", "color": "#c44e52", "lw": 1.0 + 3.5 * weight / strongest,
                                "shrinkA": 8, "shrinkB": 8, "connectionstyle": "arc3,rad=0.12"})

    ax.scatter([0], [0], s=2600, color="#dd8452", zorder=3)
    ax.text(0, 0, class_name, ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)
    for j, name in enumerate(feature_names):
        x_pos, y_pos = positions[name]
        color = "#55a868" if model.parents_[j] is None else "#4c72b0"
        ax.scatter([x_pos], [y_pos], s=260, color=color, zorder=3)
        horizontal = "left" if x_pos > 0.1 else "right" if x_pos < -0.1 else "center"
        vertical = "bottom" if y_pos > 0.9 else "top" if y_pos < -0.9 else "center"
        ax.text(x_pos * 1.08, y_pos * 1.08, name, ha=horizontal, va=vertical, fontsize=9)

    ax.plot([], [], color="#c8c8c8", linewidth=2, label=f"{class_name} → semua fitur")
    ax.plot([], [], color="#c44e52", linewidth=2.5, label="induk fitur → anak (tebal = I(A; B | kelas)), hijau = akar pohon")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=8, frameon=False)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    return ax

def plot_naive_bayes_graph(X, y, feature_names, class_name="kelas", top_pairs=8, ax=None, title=None):
    # garis abu-abu: kelas -> fitur (struktur Naive Bayes), tebal = I(fitur; kelas)
    # garis merah putus-putus: pasangan fitur yang masih bergantung setelah kelas diketahui
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