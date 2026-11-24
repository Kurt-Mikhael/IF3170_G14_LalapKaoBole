from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, writers
import numpy as np


def _log_loss_at(parameters, x, y, counts):
    losses = np.empty(len(parameters))
    for start in range(0, len(parameters), 64):
        theta = parameters[start:start + 64]
        scores = theta[:, :1] + theta[:, 1:] * x[None, :]
        losses[start:start + 64] = (
            np.logaddexp(0, (1 - 2 * y)[None, :] * scores) @ counts
        ) / counts.sum()
    return losses


def create_logreg_training_animation(
    model, X, y, *, feature_name="standardized feature", grid_size=80, fps=20,
):
    for name, value, minimum in (("grid_size", grid_size, 3), ("fps", fps, 1)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    model._check_is_fitted()
    X = model._features(X)
    y = np.asarray(y)
    if model.n_features_in_ != 1 or X.shape[1] != 1:
        raise ValueError("A theta_0/theta_1 contour requires a one-feature model")
    if y.ndim != 1 or len(y) != len(X) or not np.isin(y, model.classes_).all():
        raise ValueError("y must match X and contain only the fitted classes")
    if model.penalty is not None and model.reg_strength > 0:
        raise ValueError("Use penalty=None to visualize the trained log-loss objective")
    if not model.parameter_history_:
        raise ValueError("Fit the model with record_history=True before animating")
    trajectory = np.array([[b, w[0]] for w, b in model.parameter_history_])
    if len(trajectory) != len(model.loss_history_) or not np.isfinite(trajectory).all():
        raise ValueError("Parameter history must match the recorded training epochs")

    # pasangan (fitur, label) yang sama digabung, loss data tetap sama persis
    pairs, counts = np.unique(
        np.column_stack((X[:, 0], (y == model.classes_[1]).astype(float))),
        axis=0, return_counts=True,
    )
    x, encoded = pairs.T
    losses = _log_loss_at(trajectory, x, encoded, counts)
    if not np.allclose(losses, model.loss_history_, rtol=1e-7, atol=1e-9):
        raise ValueError("Supply the same X/y used to train the recorded model")

    low, high = trajectory.min(axis=0), trajectory.max(axis=0)
    padding = np.maximum((high - low) * .2, .35)
    theta0 = np.linspace(low[0] - padding[0], high[0] + padding[0], grid_size)
    theta1 = np.linspace(low[1] - padding[1], high[1] + padding[1], grid_size)
    theta0_grid, theta1_grid = np.meshgrid(theta0, theta1)
    grid = np.column_stack((theta0_grid.ravel(), theta1_grid.ravel()))
    surface = _log_loss_at(grid, x, encoded, counts).reshape(theta0_grid.shape)

    fig, (contour_ax, loss_ax) = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    fig.suptitle("Logistic regression | Gradient ascent", fontsize=16)
    filled = contour_ax.contourf(theta0_grid, theta1_grid, surface, levels=20, cmap="Blues")
    lines = contour_ax.contour(
        theta0_grid, theta1_grid, surface, levels=filled.levels[::2],
        colors="#65778a", linewidths=.6,
    )
    contour_ax.clabel(lines, inline=True, fontsize=8, fmt="%.3f")
    fig.colorbar(filled, ax=contour_ax, label="Mean log-loss")
    contour_ax.set(
        xlabel=r"$\theta_0$ (bias)", ylabel=r"$\theta_1$ (weight)",
        title=f"Log-loss contours | {feature_name}",
    )
    contour_ax.plot(*trajectory[0], "o", color="#238343", label="Start", zorder=4)
    path_line, = contour_ax.plot([], [], color="#c92a53", lw=2, label="Parameter path")
    current, = contour_ax.plot([], [], "o", color="#e67700", markersize=8, zorder=5, label="Current")
    contour_ax.legend(loc="best", fontsize=8)

    epochs = np.arange(len(trajectory))
    loss_padding = max(float(np.ptp(losses)) * .15, .01)
    loss_ax.set(
        xlabel="Epoch", ylabel="Mean log-loss",
        xlim=(0, max(1, epochs[-1])),
        ylim=(max(0, losses.min() - loss_padding), losses.max() + loss_padding),
        title="Log-likelihood rises; log-loss falls",
    )
    loss_ax.grid(alpha=.2)
    loss_line, = loss_ax.plot([], [], color="#176b87", lw=2)
    loss_point, = loss_ax.plot([], [], "o", color="#e67700")
    status = loss_ax.text(.04, .95, "", transform=loss_ax.transAxes, va="top", fontsize=10)

    def update(epoch):
        path_line.set_data(trajectory[:epoch + 1, 0], trajectory[:epoch + 1, 1])
        current.set_data([trajectory[epoch, 0]], [trajectory[epoch, 1]])
        loss_line.set_data(epochs[:epoch + 1], losses[:epoch + 1])
        loss_point.set_data([epoch], [losses[epoch]])
        status.set_text(
            f"Epoch {epoch}/{epochs[-1]}\n"
            f"theta_0 = {trajectory[epoch, 0]:.4f}\n"
            f"theta_1 = {trajectory[epoch, 1]:.4f}\n"
            f"Log-loss = {losses[epoch]:.6f}"
        )
        return path_line, current, loss_line, loss_point, status

    # keadaan awal dan akhir ditahan agar video mudah diperiksa
    frames = np.concatenate((np.zeros(fps, dtype=int), epochs, np.full(2 * fps, epochs[-1])))
    return FuncAnimation(
        fig, update, frames=frames, interval=1000 / fps,
        init_func=lambda: update(0), repeat=False,
    )


def _ffmpeg_executable():
    if writers.is_available("ffmpeg"):
        return mpl.rcParams["animation.ffmpeg_path"]
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "MP4 export requires FFmpeg. Install imageio-ffmpeg with "
            "'python -m pip install imageio-ffmpeg', or add FFmpeg to PATH."
        ) from exc


def save_logreg_training_video(
    model, X, y, path, *, feature_name="standardized feature", grid_size=80, fps=20, dpi=100,
):
    destination = Path(path)
    if destination.suffix.lower() != ".mp4":
        raise ValueError("The video destination must have an .mp4 extension")
    ffmpeg = _ffmpeg_executable()
    with mpl.rc_context({"animation.ffmpeg_path": ffmpeg}):
        video = create_logreg_training_animation(
            model, X, y, feature_name=feature_name, grid_size=grid_size, fps=fps,
        )
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            writer = FFMpegWriter(
                fps=fps, codec="libx264",
                extra_args=[
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-threads", "1", "-preset", "veryfast",
                ],
            )
            video.save(str(destination), writer=writer, dpi=dpi)
        finally:
            plt.close(video._fig)
    return destination
