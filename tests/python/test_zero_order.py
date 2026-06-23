from __future__ import annotations

import numpy as np

import optlib


def test_zero_order_methods_accept_python_value_function() -> None:
    target = np.array([1.5, -2.0])

    def value(values: np.ndarray) -> float:
        shifted = values - target
        return float(np.dot(shifted, shifted))

    for method in ["nelder_mead", "powell", "coordinate_search"]:
        result = optlib.MinimizeZeroOrder(
            value,
            np.array([-2.0, 3.0]),
            method=method,
            max_iter=400,
            step_tolerance=1e-7,
            function_tolerance=1e-12,
            initial_step=0.8,
            line_search_radius=3.0,
            log_trajectory=True,
        )
        assert result["value"] < 1e-6
        assert result["function_evaluations"] > 0
        assert result["trajectory"]["x"].shape[1] == 2


def test_zero_order_methods_reduce_rosenbrock() -> None:
    for method in ["nelder_mead", "powell", "coordinate_search"]:
        result = optlib.MinimizeRosenbrockZeroOrder(
            np.array([-1.2, 1.0]),
            method=method,
            max_iter=600,
            step_tolerance=1e-7,
            function_tolerance=1e-12,
            initial_step=0.8,
            line_search_radius=2.0,
            log_trajectory=False,
        )
        assert result["value"] < 1.0
