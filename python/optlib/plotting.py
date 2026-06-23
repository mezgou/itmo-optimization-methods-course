"""Plotting helpers for Lab 2 notebooks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


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
) -> object:
    """Draw contours and an optional optimization trajectory."""

    grid_x, grid_y, grid_z = objective_grid(
        value_function,
        x_limits=x_limits,
        y_limits=y_limits,
    )
    levels = np.geomspace(
        max(float(np.nanmin(grid_z)), 1e-6), max(float(np.nanmax(grid_z)), 1e-5), 24
    )
    contour = axis.contour(grid_x, grid_y, grid_z, levels=levels, cmap="viridis")
    if trajectory is not None and len(trajectory) > 0:
        axis.plot(trajectory[:, 0], trajectory[:, 1], marker=".", linewidth=1.0, markersize=2)
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(True, alpha=0.25)
    return contour
