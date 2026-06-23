"""Reusable, consistently styled plotting helpers for the lab notebooks.

Matplotlib is imported lazily inside each function so that ``import optlib``
never requires a plotting backend. All helpers accept an existing ``axis`` (or
``figure``) so notebooks stay in control of the overall layout.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

# Compact categorical palette reused across every notebook.
PALETTE: tuple[str, ...] = (
    "#2563eb",  # blue
    "#059669",  # green
    "#dc2626",  # red
    "#7c3aed",  # violet
    "#d97706",  # amber
    "#0891b2",  # cyan
    "#be123c",  # rose
    "#4b5563",  # gray
)


def use_notebook_style() -> None:
    """Apply one consistent, polished matplotlib style for all notebooks."""

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (9.5, 5.2),
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "figure.autolayout": True,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=list(PALETTE)),
            "image.cmap": "viridis",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.45,
            "lines.markersize": 4,
        }
    )


def objective_grid(
    value_function: Callable[[np.ndarray], float],
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a 2D objective on a regular mesh."""

    x_values = np.linspace(x_limits[0], x_limits[1], points)
    y_values = np.linspace(y_limits[0], y_limits[1], points)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid_z = np.empty_like(grid_x)
    for row in range(points):
        for col in range(points):
            grid_z[row, col] = value_function(np.array([grid_x[row, col], grid_y[row, col]]))
    return grid_x, grid_y, grid_z


def plot_contours(
    axis: object,
    value_function: Callable[[np.ndarray], float],
    *,
    x_limits: tuple[float, float] = (-5.0, 5.0),
    y_limits: tuple[float, float] = (-5.0, 5.0),
    trajectory: np.ndarray | None = None,
    title: str = "",
    points: int = 200,
    log_levels: bool = True,
) -> object:
    """Draw contours and an optional single optimization trajectory."""

    grid_x, grid_y, grid_z = objective_grid(
        value_function, x_limits=x_limits, y_limits=y_limits, points=points
    )
    if log_levels:
        low = max(float(np.nanmin(grid_z)), 1e-6)
        high = max(float(np.nanmax(grid_z)), low * 10.0)
        levels = np.geomspace(low, high, 24)
    else:
        levels = 24
    contour = axis.contourf(grid_x, grid_y, grid_z, levels=levels, cmap="viridis", alpha=0.82)
    axis.contour(grid_x, grid_y, grid_z, levels=levels, colors="white", linewidths=0.25, alpha=0.32)
    if trajectory is not None and len(trajectory) > 0:
        trajectory = np.asarray(trajectory, dtype=np.float64)
        axis.plot(trajectory[:, 0], trajectory[:, 1], color="#dc2626", linewidth=1.2, marker=".")
        axis.plot(trajectory[0, 0], trajectory[0, 1], "o", color="white", markersize=5, mec="black")
        axis.plot(
            trajectory[-1, 0], trajectory[-1, 1], "*", color="#f59e0b", markersize=12, mec="black"
        )
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    return contour


def plot_trajectories(
    axis: object,
    value_function: Callable[[np.ndarray], float],
    results: Mapping[str, Mapping[str, Any]],
    *,
    x_limits: tuple[float, float] = (-2.0, 2.0),
    y_limits: tuple[float, float] = (-1.0, 3.0),
    title: str = "",
    points: int = 200,
    minimum: Sequence[float] | None = None,
) -> object:
    """Draw shared contours plus one labelled trajectory per method result."""

    grid_x, grid_y, grid_z = objective_grid(
        value_function, x_limits=x_limits, y_limits=y_limits, points=points
    )
    low = max(float(np.nanmin(grid_z)), 1e-6)
    high = max(float(np.nanmax(grid_z)), low * 10.0)
    levels = np.geomspace(low, high, 24)
    contour = axis.contourf(grid_x, grid_y, grid_z, levels=levels, cmap="viridis", alpha=0.8)
    axis.contour(grid_x, grid_y, grid_z, levels=levels, colors="white", linewidths=0.22, alpha=0.3)
    for index, (name, result) in enumerate(results.items()):
        path = np.asarray(result["trajectory"]["x"], dtype=np.float64)
        color = PALETTE[index % len(PALETTE)]
        axis.plot(path[:, 0], path[:, 1], color=color, linewidth=1.3, label=name, marker="")
        axis.plot(path[0, 0], path[0, 1], "o", color=color, markersize=4, mec="black")
    if minimum is not None:
        axis.plot(
            minimum[0],
            minimum[1],
            "*",
            color="#f59e0b",
            markersize=13,
            mec="black",
            label="минимум",
        )
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.legend(loc="upper right")
    return contour


def plot_convergence(
    axis: object,
    results: Mapping[str, Mapping[str, Any]],
    *,
    key: str = "f",
    title: str = "",
    ylabel: str | None = None,
) -> None:
    """Plot f(x) or gradient norm versus iteration on a log scale per method."""

    for index, (name, result) in enumerate(results.items()):
        series = np.asarray(result["trajectory"][key], dtype=np.float64)
        series = np.clip(series, 1e-16, None)
        axis.semilogy(
            np.arange(series.size), series, color=PALETTE[index % len(PALETTE)], label=name
        )
    axis.set_title(title)
    axis.set_xlabel("итерация")
    axis.set_ylabel(ylabel or ("f(x)" if key == "f" else "‖∇f(x)‖"))
    axis.legend()


def plot_surface3d(
    figure: object,
    value_function: Callable[[np.ndarray], float],
    *,
    x_limits: tuple[float, float] = (-5.0, 5.0),
    y_limits: tuple[float, float] = (-5.0, 5.0),
    points: int = 120,
    title: str = "",
    position: int = 111,
) -> object:
    """Add a styled 3D surface subplot to ``figure`` and return the axis."""

    grid_x, grid_y, grid_z = objective_grid(
        value_function, x_limits=x_limits, y_limits=y_limits, points=points
    )
    axis = figure.add_subplot(position, projection="3d")
    axis.plot_surface(
        grid_x, grid_y, grid_z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.92
    )
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    return axis


def plot_confusion_matrix(
    axis: object,
    matrix: Sequence[Sequence[int]],
    *,
    labels: Sequence[str] = ("0", "1"),
    title: str = "Confusion matrix",
) -> None:
    """Render a small annotated confusion-matrix heatmap."""

    data = np.asarray(matrix, dtype=np.int64)
    image = axis.imshow(data, cmap="Blues")
    threshold = data.max() / 2.0 if data.max() else 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            axis.text(
                j,
                i,
                str(int(data[i, j])),
                ha="center",
                va="center",
                color="white" if data[i, j] > threshold else "black",
                fontsize=11,
                fontweight="bold",
            )
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    axis.set_xlabel("предсказание")
    axis.set_ylabel("истина")
    axis.set_title(title)
    axis.grid(False)
    return image


def plot_decision_boundary(
    axis: object,
    predict_proba: Callable[[np.ndarray], np.ndarray],
    features: np.ndarray,
    targets: np.ndarray,
    *,
    title: str = "",
    points: int = 220,
    padding: float = 0.5,
) -> None:
    """Draw a 2D probability decision surface with the scattered samples."""

    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets)
    x_min, x_max = features[:, 0].min() - padding, features[:, 0].max() + padding
    y_min, y_max = features[:, 1].min() - padding, features[:, 1].max() + padding
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, points), np.linspace(y_min, y_max, points)
    )
    mesh = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    probabilities = np.asarray(predict_proba(mesh), dtype=np.float64).reshape(grid_x.shape)
    axis.contourf(grid_x, grid_y, probabilities, levels=18, cmap="RdBu_r", alpha=0.68)
    axis.contour(grid_x, grid_y, probabilities, levels=[0.5], colors="black", linewidths=1.0)
    for label, color in ((0, "#2563eb"), (1, "#dc2626")):
        mask = targets.astype(np.int64) == label
        axis.scatter(
            features[mask, 0],
            features[mask, 1],
            s=12,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
    axis.set_title(title)
    axis.set_xlabel("признак 0")
    axis.set_ylabel("признак 1")
    axis.grid(False)


def bar_comparison(
    axis: object,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str = "",
    ylabel: str = "",
    annotate: bool = False,
    highlight_max: bool = True,
) -> None:
    """Render a styled bar chart, optionally highlighting the best bar."""

    labels = list(labels)
    values = np.asarray(values, dtype=np.float64)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    if highlight_max and values.size:
        best = int(np.nanargmax(values))
        colors[best] = "#16a34a"
    bars = axis.bar(labels, values, color=colors)
    if annotate:
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=30)
