"""Small reproducible harness for Lab 2 optimizer comparisons."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter
from typing import Any

import numpy as np

from ._optlib import (
    MinimizeBenchmarkFunction,
    MinimizeBenchmarkFunctionSecondOrder,
    MinimizeBenchmarkFunctionZeroOrder,
    MinimizeRosenbrock,
    MinimizeRosenbrockSecondOrder,
    MinimizeRosenbrockZeroOrder,
)
from .functions import Objective

FIRST_ORDER_METHODS = {
    "gradient_descent",
    "gd",
    "heavy_ball",
    "nesterov",
    "adam",
    "rmsprop",
    "adagrad",
}
SECOND_ORDER_METHODS = {"newton", "bfgs", "lbfgs", "l_bfgs"}
ZERO_ORDER_METHODS = {"nelder_mead", "powell", "coordinate_search", "coordinate_descent"}


def _normalize_method(method: str) -> str:
    return method.lower().replace("-", "_")


def result_summary(
    method: str,
    result: dict[str, Any],
    objective: Objective | None = None,
    wall_ms: float | None = None,
) -> dict[str, Any]:
    """Convert an optimizer result into a compact table row."""

    x_value = np.asarray(result["x"], dtype=np.float64)
    distance = np.nan
    if objective is not None and objective.minimum is not None:
        distance = float(np.linalg.norm(x_value - objective.minimum))
    return {
        "objective": None if objective is None else objective.name,
        "method": method,
        "value": float(result["value"]),
        "gradient_norm": float(result.get("gradient_norm", np.nan)),
        "iterations": int(result["iterations"]),
        "function_evaluations": int(result.get("function_evaluations", 0)),
        "gradient_evaluations": int(result.get("gradient_evaluations", 0)),
        "hessian_evaluations": int(result.get("hessian_evaluations", 0)),
        "wall_ms": np.nan if wall_ms is None else wall_ms,
        "converged": bool(result["converged"]),
        "distance_to_minimum": distance,
    }


def _run_native_rosenbrock(
    method: str,
    method_class: str,
    x0: np.ndarray,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    if method_class == "first":
        return MinimizeRosenbrock(x0, method=method, **kwargs)
    if method_class == "second":
        return MinimizeRosenbrockSecondOrder(x0, method=method, **kwargs)
    return MinimizeRosenbrockZeroOrder(x0, method=method, **kwargs)


def _run_native_benchmark(
    objective: Objective,
    method: str,
    method_class: str,
    x0: np.ndarray,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    if objective.native_name is None:
        msg = f"{objective.name} has no native benchmark wrapper"
        raise ValueError(msg)
    if method_class == "first":
        return MinimizeBenchmarkFunction(
            objective.native_name,
            x0,
            method=method,
            scale=objective.scale,
            **kwargs,
        )
    if method_class == "second":
        return MinimizeBenchmarkFunctionSecondOrder(
            objective.native_name, x0, method=method, scale=objective.scale, **kwargs
        )
    return MinimizeBenchmarkFunctionZeroOrder(
        objective.native_name, x0, method=method, scale=objective.scale, **kwargs
    )


def run_method(
    objective: Objective,
    x0: np.ndarray,
    method: str,
    *,
    max_iter: int = 1000,
    gradient_tolerance: float = 1e-6,
    step_tolerance: float = 1e-10,
    function_tolerance: float = 1e-12,
    log_trajectory: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one optlib optimizer against an Objective descriptor."""

    normalized = _normalize_method(method)
    start = np.asarray(x0, dtype=np.float64)
    common_kwargs = {
        "max_iter": max_iter,
        "step_tolerance": step_tolerance,
        "function_tolerance": function_tolerance,
        "log_trajectory": log_trajectory,
        **kwargs,
    }

    if normalized in FIRST_ORDER_METHODS:
        if objective.gradient is None:
            msg = f"{objective.name} does not provide a gradient for first-order optimization"
            raise ValueError(msg)
        common_kwargs["gradient_tolerance"] = gradient_tolerance
        if objective.name == "rosenbrock":
            return _run_native_rosenbrock(method, "first", start, common_kwargs)
        return _run_native_benchmark(objective, method, "first", start, common_kwargs)

    if normalized in SECOND_ORDER_METHODS:
        if objective.gradient is None:
            msg = f"{objective.name} does not provide a gradient for second-order optimization"
            raise ValueError(msg)
        common_kwargs["gradient_tolerance"] = gradient_tolerance
        if objective.name == "rosenbrock":
            return _run_native_rosenbrock(method, "second", start, common_kwargs)
        return _run_native_benchmark(objective, method, "second", start, common_kwargs)

    if normalized in ZERO_ORDER_METHODS:
        if objective.name == "rosenbrock":
            return _run_native_rosenbrock(method, "zero", start, common_kwargs)
        return _run_native_benchmark(objective, method, "zero", start, common_kwargs)

    msg = f"Unknown optimizer method: {method}"
    raise ValueError(msg)


def compare_methods(
    objective: Objective,
    x0: np.ndarray,
    methods: Iterable[str],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run several methods and return compact result rows."""

    rows: list[dict[str, Any]] = []
    for method in methods:
        started_at = perf_counter()
        result = run_method(objective, x0, method, **kwargs)
        wall_ms = (perf_counter() - started_at) * 1000.0
        rows.append(result_summary(method, result, objective, wall_ms))
    return rows


def multistart_compare(
    objective: Objective,
    methods: Iterable[str],
    *,
    starts: np.ndarray,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run methods over deterministic starts and return one row per run."""

    rows: list[dict[str, Any]] = []
    for start_index, start in enumerate(np.asarray(starts, dtype=np.float64)):
        for row in compare_methods(objective, start, methods, **kwargs):
            row["start_index"] = start_index
            rows.append(row)
    return rows


def scipy_minimize(
    objective: Objective,
    x0: np.ndarray,
    *,
    method: str = "BFGS",
    max_iter: int = 1000,
) -> dict[str, Any] | None:
    """Run scipy.optimize.minimize when scipy is installed."""

    try:
        from scipy import optimize
    except ModuleNotFoundError:
        return None

    started_at = perf_counter()
    options = {"maxiter": max_iter}
    jac = objective.gradient if objective.gradient is not None else None
    hess = objective.hessian if method.lower() in {"newton-cg", "trust-ncg"} else None
    result = optimize.minimize(
        objective.value,
        np.asarray(x0, dtype=np.float64),
        method=method,
        jac=jac,
        hess=hess,
        options=options,
    )
    wall_ms = (perf_counter() - started_at) * 1000.0
    jacobian = getattr(result, "jac", None)
    return {
        "objective": objective.name,
        "method": f"scipy:{method}",
        "value": float(result.fun),
        "gradient_norm": float(np.linalg.norm(jacobian)) if jacobian is not None else np.nan,
        "iterations": int(getattr(result, "nit", 0)),
        "function_evaluations": int(getattr(result, "nfev", 0)),
        "gradient_evaluations": int(getattr(result, "njev", 0)),
        "hessian_evaluations": int(getattr(result, "nhev", 0)),
        "wall_ms": wall_ms,
        "converged": bool(result.success),
        "distance_to_minimum": float(np.linalg.norm(result.x - objective.minimum))
        if objective.minimum is not None
        else np.nan,
    }
