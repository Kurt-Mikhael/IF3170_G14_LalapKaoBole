import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


class DrawNode:
    # node yang benar-benar digambar (hasil pemotongan top_n / max_depth)
    def __init__(self, label, kind, prediction=None):
        self.label = label
        self.kind = kind              
        self.children = []            # list of (edge_label, DrawNode)
        self.x = 0.0
        self.depth = 0


def format_number(value):
    return f"{value:.4g}"


def edge_label(node, key, show_probability):
    if node.threshold is None:
        label = str(key)
    elif key == "left":
        label = f"<= {format_number(node.threshold)}"
    else:
        label = f"> {format_number(node.threshold)}"
    if show_probability:
        label += f"\n(p={node.probabilities[key]:.2f})"
    return label


def build(model, node, depth, top_n, max_depth, show_probability):
    confidence = node.distribution[node.prediction]
    class_name = model.classes[node.prediction]

    if node.is_leaf():
        return DrawNode(f"{class_name}\n({confidence:.0%})", "leaf", node.prediction)

    feature = model.feature_names[node.feature]
    if max_depth is not None and depth >= max_depth:
        return DrawNode(f"{feature}\n...\nmayoritas: {class_name}", "truncated", node.prediction)

    draw = DrawNode(feature, "internal", node.prediction)
    # cabang diurutkan berdasarkan p_i 
    ranked = sorted(node.children, key=lambda k: node.probabilities[k], reverse=True)
    shown = ranked if top_n is None else ranked[:top_n]
    if node.threshold is not None:
        shown = [k for k in ("left", "right") if k in shown]  # numerik: tetap urut <= lalu >

    for key in shown:
        child = build(model, node.children[key], depth + 1, top_n, max_depth, show_probability)
        draw.children.append((edge_label(node, key, show_probability), child))

    hidden = len(ranked) - len(shown)
    if hidden > 0:
        rest = sum(node.probabilities[k] for k in ranked if k not in shown)
        label = f"+{hidden} cabang lain"
        if show_probability:
            label += f"\n(p={rest:.2f})"
        draw.children.append(("...", DrawNode(label, "hidden")))
    return draw


def layout(node, depth, next_x):
    # leaf diberi posisi x berurutan; node internal di tengah anak-anaknya
    node.depth = depth
    if not node.children:
        node.x = next_x
        return next_x + 1
    for _, child in node.children:
        next_x = layout(child, depth + 1, next_x)
    node.x = (node.children[0][1].x + node.children[-1][1].x) / 2
    return next_x


def iter_nodes(node):
    yield node
    for _, child in node.children:
        yield from iter_nodes(child)


def plot_tree(model, top_n=None, max_depth=None, *, path=None, show_probability=True,title=None, figsize=None, dpi=150, fontsize=8):
    if not hasattr(model, "root"):
        raise ValueError("model belum di-fit")
    for name, value in (("top_n", top_n), ("max_depth", max_depth)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == "top_n" else 0)):
            raise ValueError(f"{name} harus None atau integer {'>= 1' if name == 'top_n' else '>= 0'}")

    root = build(model, model.root, 0, top_n, max_depth, show_probability)
    n_leaves = layout(root, 0, 0)
    nodes = list(iter_nodes(root))
    depth = max(n.depth for n in nodes)

    if figsize is None:
        figsize = (min(max(6, n_leaves * 1.4), 200), max(3, (depth + 1) * 1.6))
    fig, ax = plt.subplots(figsize=figsize)

    colors = {i: PALETTE[i % len(PALETTE)] for i in range(len(model.classes))}
    box = dict(boxstyle="round,pad=0.4", linewidth=1)

    for node in nodes:
        for branch_text, child in node.children:
            ax.plot([node.x, child.x], [-node.depth, -child.depth], color="#999999", lw=1, zorder=1,
                    linestyle=":" if child.kind == "hidden" else "-")
            ax.text((node.x + child.x) / 2, -(node.depth + child.depth) / 2, branch_text,
                    ha="center", va="center", fontsize=fontsize - 1, zorder=2,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

        style = dict(box)
        if node.kind == "leaf":
            style.update(fc=colors[node.prediction], ec="black", alpha=0.85)
            text_color = "white"
        elif node.kind == "internal":
            style.update(fc="#F2F2F2", ec="black")
            text_color = "black"
        elif node.kind == "truncated":
            style.update(fc="white", ec=colors[node.prediction], linestyle="--", linewidth=1.5)
            text_color = "black"
        else:  # hidden
            style.update(fc="white", ec="#999999", linestyle=":")
            text_color = "#666666"
        ax.text(node.x, -node.depth, node.label, ha="center", va="center", fontsize=fontsize,
                color=text_color, bbox=style, zorder=3,
                fontweight="bold" if node.kind == "internal" else "normal")

    ax.set_xlim(-0.7, n_leaves - 0.3)
    ax.set_ylim(-depth - 0.6, 0.6)
    ax.axis("off")

    handles = [Patch(fc=colors[i], ec="black", label=str(c)) for i, c in enumerate(model.classes)]
    ax.legend(handles=handles, title="Kelas (leaf)", loc="upper right", fontsize=fontsize, frameon=False)
    if title is None:
        parts = [f"C4.5 Decision Tree ({model.count_nodes()} node)"]
        if top_n is not None:
            parts.append(f"top-{top_n} cabang/node")
        if max_depth is not None:
            parts.append(f"kedalaman <= {max_depth}")
        title = " | ".join(parts)
    ax.set_title(title)
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig, ax
