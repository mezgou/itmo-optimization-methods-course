from __future__ import annotations

import numpy as np

import optlib


def test_minimize_rosenbrock_returns_result_and_trajectory() -> None:
    result = optlib.MinimizeRosenbrock(
        np.array([1.2, 1.2]),
        method="adam",
        learning_rate=1e-2,
        max_iter=30000,
        gradient_tolerance=1e-4,
        step_tolerance=0.0,
        function_tolerance=0.0,
    )

    assert result["value"] < 1e-4
    assert result["iterations"] > 0
    assert result["trajectory"]["x"].shape[1] == 2
    assert result["trajectory"]["f"].shape[0] == result["trajectory"]["x"].shape[0]


def test_general_minimize_accepts_python_callables() -> None:
    target = np.array([1.0, -2.0, 0.5])

    def value(values: np.ndarray) -> float:
        shifted = values - target
        return float(np.dot(shifted, shifted))

    def gradient(values: np.ndarray) -> np.ndarray:
        return 2.0 * (values - target)

    result = optlib.Minimize(
        value,
        gradient,
        np.array([4.0, 4.0, 4.0]),
        method="gradient_descent",
        learning_rate=0.1,
        max_iter=1000,
        gradient_tolerance=1e-8,
        step_tolerance=0.0,
        function_tolerance=0.0,
    )

    assert result["converged"]
    np.testing.assert_allclose(result["x"], target, atol=1e-6)
    assert result["value"] < 1e-12
