from __future__ import annotations

import numpy as np
import pytest

import optlib


def test_rosenbrock_matches_scipy_bfgs_reference() -> None:
    scipy_optimize = pytest.importorskip("scipy.optimize")

    initial_x = np.array([-1.2, 1.0])
    scipy_result = scipy_optimize.minimize(
        lambda values: optlib.RosenbrockValue(np.asarray(values, dtype=np.float64)),
        initial_x,
        jac=lambda values: optlib.RosenbrockGradient(np.asarray(values, dtype=np.float64)),
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 20_000},
    )
    optlib_result = optlib.MinimizeRosenbrock(
        initial_x,
        method="adam",
        learning_rate=2e-2,
        max_iter=30_000,
        gradient_tolerance=1e-6,
        step_tolerance=0.0,
        function_tolerance=0.0,
        log_trajectory=False,
    )

    assert scipy_result.success
    assert optlib_result["value"] < 1e-5
    assert scipy_result.fun < 1e-10
    np.testing.assert_allclose(optlib_result["x"], scipy_result.x, atol=2e-3)
