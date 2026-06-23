from __future__ import annotations

import numpy as np

import optlib


def test_newton_accepts_python_hessian() -> None:
    target = np.array([2.0, -1.0, 0.5])

    def value(values: np.ndarray) -> float:
        shifted = values - target
        return float(0.5 * np.dot(shifted, shifted))

    def gradient(values: np.ndarray) -> np.ndarray:
        return values - target

    def hessian(values: np.ndarray) -> np.ndarray:
        return np.eye(values.shape[0])

    result = optlib.MinimizeSecondOrder(
        value,
        gradient,
        np.array([-3.0, 4.0, 5.0]),
        method="newton",
        hessian_function=hessian,
        max_iter=20,
        gradient_tolerance=1e-8,
        step_tolerance=0.0,
        function_tolerance=0.0,
    )

    assert result["converged"]
    np.testing.assert_allclose(result["x"], target, atol=1e-7)
    assert result["value"] < 1e-12


def test_bfgs_and_lbfgs_rosenbrock() -> None:
    for method in ["bfgs", "lbfgs"]:
        result = optlib.MinimizeRosenbrockSecondOrder(
            np.array([-1.2, 1.0]),
            method=method,
            max_iter=1000,
            gradient_tolerance=1e-5,
            step_tolerance=0.0,
            function_tolerance=0.0,
            log_trajectory=True,
        )
        assert result["value"] < 1e-8
        assert result["gradient_norm"] < 1e-4
        assert result["trajectory"]["x"].shape[1] == 2


def test_lbfgs_rosenbrock_100d() -> None:
    start = np.empty(100)
    start[0::2] = -1.2
    start[1::2] = 1.0
    result = optlib.MinimizeRosenbrockSecondOrder(
        start,
        method="lbfgs",
        max_iter=3000,
        gradient_tolerance=1e-4,
        step_tolerance=0.0,
        function_tolerance=0.0,
        history_size=12,
        log_trajectory=False,
    )
    assert result["value"] < 1e-4
    assert result["gradient_norm"] < 1e-2
