from __future__ import annotations

import numpy as np

import optlib


def test_scheduler_keywords_keep_first_order_working() -> None:
    result = optlib.MinimizeRosenbrock(
        np.array([1.2, 1.2]),
        method="adam",
        learning_rate=2e-2,
        max_iter=30000,
        gradient_tolerance=1e-4,
        step_tolerance=0.0,
        function_tolerance=0.0,
        schedule="cosine",
        minimum_learning_rate=1e-4,
        schedule_iterations=30000,
        warmup_steps=5,
        log_trajectory=False,
    )
    assert result["value"] < 1e-4


def test_second_order_bindings_converge() -> None:
    for method in ["newton", "bfgs", "lbfgs"]:
        result = optlib.MinimizeRosenbrockSecondOrder(
            np.array([-1.2, 1.0]),
            method=method,
            max_iter=1000,
            gradient_tolerance=1e-5,
            step_tolerance=0.0,
            function_tolerance=0.0,
        )
        assert result["value"] < 1e-8
        assert result["gradient_norm"] < 1e-4


def test_zero_order_bindings_reduce_rosenbrock() -> None:
    for method in ["nelder_mead", "powell", "coordinate_descent"]:
        result = optlib.MinimizeRosenbrockZeroOrder(
            np.array([1.2, 1.2]),
            method=method,
            max_iter=2000,
            step_tolerance=1e-7,
            function_tolerance=1e-10,
            initial_step=0.5,
        )
        assert result["value"] < 1e-3
