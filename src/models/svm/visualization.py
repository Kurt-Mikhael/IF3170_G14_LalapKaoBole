"""Generic plotting helpers for fitted two-feature decision models."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_svm_boundary(model, X, y, *, ax=None, title=None, resolution=150):
    """Plot samples, support vectors, and available decision-function contours.

    Contours show decision-function levels, not Euclidean distances (especially
    for nonlinear kernels). The model must already be fitted on two features.
    """
    try:
        X = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("X must contain finite numeric features") from exc
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != 2 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite array with shape (n_samples, 2)")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must be one-dimensional and match the number of samples")
    if (
        isinstance(resolution, (bool, np.bool_))
        or not isinstance(resolution, (int, np.integer))
        or resolution < 2
    ):
        raise ValueError("resolution must be an integer of at least 2")

    try:
        classes = np.asarray(model.classes_)
        support = np.asarray(model.support_vectors_, dtype=float)
        feature_count = model.n_features_in_
        decision_function = model.decision_function
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            "model must be fitted and expose classes_, support_vectors_, n_features_in_, and decision_function"
        ) from exc
    if classes.ndim != 1 or len(classes) != 2:
        raise ValueError("model must have exactly two fitted classes")
    if feature_count != 2:
        raise ValueError("model must be fitted with exactly two features")
    if support.ndim != 2 or support.shape[1] != 2 or not np.isfinite(support).all():
        raise ValueError(
            "model support_vectors_ must be a finite array with shape (n_support, 2)"
        )
    if not callable(decision_function):
        raise TypeError("model must be fitted and expose decision_function")
    if not np.isin(y, classes).all():
        raise ValueError("y contains labels not present in model.classes_")

    # Bounds include both observed samples and support vectors, then add modest padding.
    points = np.concatenate((X, support), axis=0)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = upper - lower
    span[span == 0] = 1.0
    lower = lower - 0.05 * span
    upper = upper + 0.05 * span
    # x_grid and y_grid define feature-coordinate axes shared by contour evaluation.
    x_grid = np.linspace(lower[0], upper[0], resolution)
    y_grid = np.linspace(lower[1], upper[1], resolution)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    score_chunks = []
    for start in range(0, len(grid_points), 512):
        try:
            scores = np.asarray(decision_function(grid_points[start : start + 512]))
        except RuntimeError as exc:
            raise ValueError("model must be fitted before plotting") from exc
        if (
            scores.ndim != 1
            or len(scores) != len(grid_points[start : start + 512])
            or not np.isfinite(scores).all()
        ):
            raise ValueError(
                "decision_function must return one finite score per input row"
            )
        score_chunks.append(scores)
    score_grid = np.concatenate(score_chunks).reshape(grid_x.shape)

    if ax is None:
        _, ax = plt.subplots()
    markers = ("o", "^", "s", "D")
    colors = ("#0072B2", "#D55E00")
    for index, label in enumerate(classes):
        mask = y == label
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            color=colors[index],
            marker=markers[index],
            label=str(label),
            zorder=3,
        )
    if len(support):
        ax.scatter(
            support[:, 0],
            support[:, 1],
            facecolors="none",
            edgecolors="black",
            marker="o",
            s=100,
            linewidths=1.5,
            label="Support vectors",
            zorder=4,
        )

    level_styles: dict[int, Literal["--", "-"]] = {-1: "--", 0: "-", 1: "--"}
    missing = []
    for level in (-1, 0, 1):
        if (
            score_grid.min() <= level <= score_grid.max()
            and score_grid.min() != score_grid.max()
        ):
            contour = ax.contour(
                grid_x,
                grid_y,
                score_grid,
                levels=[level],
                colors="black",
                linestyles=level_styles[level],
                linewidths=1.2,
            )
            if not any(np.asarray(path.vertices).shape[0] > 1 for path in contour.get_paths()):
                contour.remove()
                missing.append(level)
        else:
            missing.append(level)
    if missing:
        missing_labels = (f"f = {level:+d}" if level else "f = 0" for level in missing)
        if score_grid.min() == score_grid.max():
            message = f"No contours: constant decision score f = {score_grid.min():g}"
        else:
            message = "Contour(s) absent on this grid: " + ", ".join(
                missing_labels
            )
        ax.text(
            0.02,
            0.98,
            message,
            transform=ax.transAxes,
            va="top",
            fontsize="small",
        )

    handles, labels = ax.get_legend_handles_labels()
    level_labels = {
        level: (f"f = {level:+d}" if level else "f = 0") for level in (-1, 0, 1)
    }
    handles.extend(
        Line2D(
            [],
            [],
            color="black",
            linestyle=level_styles[level],
            label=level_labels[level],
        )
        for level in (-1, 0, 1)
        if level not in missing
    )
    ax.legend(
        handles=handles,
        labels=labels
        + [level_labels[level] for level in (-1, 0, 1) if level not in missing],
    )
    ax.set(
        xlim=(lower[0], upper[0]),
        ylim=(lower[1], upper[1]),
        xlabel="Feature 1",
        ylabel="Feature 2",
    )
    if title is not None:
        ax.set_title(title)
    return ax
