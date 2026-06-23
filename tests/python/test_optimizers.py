from __future__ import annotations

import numpy as np

import optlib


def test_first_order_methods_reduce_rosenbrock() -> None:
    configs = {
        "gradient_descent": {"learning_rate": 1e-3},
        "heavy_ball": {"learning_rate": 8e-4, "momentum": 0.8},
        "nesterov": {"learning_rate": 8e-4, "momentum": 0.8},
        "adam": {"learning_rate": 2e-2},
        "rmsprop": {"learning_rate": 2e-4, "beta2": 0.999},
        "adagrad": {"learning_rate": 5e-2},
    }

    for method, params in configs.items():
        result = optlib.MinimizeRosenbrock(
            np.array([-1.2, 1.0]),
            method=method,
            max_iter=20_000,
            gradient_tolerance=1e-3,
            step_tolerance=0.0,
            function_tolerance=0.0,
            log_trajectory=True,
            **params,
        )
        assert result["value"] < 1e-2
        assert result["gradient_norm"] < 0.25
        assert result["x"].shape == (2,)
        assert result["trajectory"]["x"].shape[1] == 2
        assert result["trajectory"]["f"].shape[0] == result["trajectory"]["x"].shape[0]


def test_adam_rosenbrock_nd() -> None:
    result = optlib.MinimizeRosenbrock(
        np.array([-1.2, 1.0, -1.2, 1.0, -1.2]),
        method="adam",
        learning_rate=2e-2,
        max_iter=30_000,
        gradient_tolerance=1e-3,
        step_tolerance=0.0,
        function_tolerance=0.0,
        log_trajectory=False,
    )
    assert result["value"] < 1e-2
    assert result["gradient_norm"] < 0.25
