"""Objective registry used by benchmark experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._optlib import (
    BenchmarkFunctionGradient,
    BenchmarkFunctionHessian,
    BenchmarkFunctionInfo,
    BenchmarkFunctionValue,
    DesmosSurfaceNumericalGradient,
    DesmosSurfaceValue,
    RosenbrockGradient,
    RosenbrockHessian,
    RosenbrockValue,
)

ArrayFunction = Callable[[np.ndarray], float]
GradientFunction = Callable[[np.ndarray], np.ndarray]
HessianFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Objective:
    """Numerical objective with optional derivative callables."""

    name: str
    dimension: int
    value: ArrayFunction
    gradient: GradientFunction | None = None
    hessian: HessianFunction | None = None
    minimum: np.ndarray | None = None
    minimum_value: float = 0.0
    derivative_free: bool = False
    native_name: str | None = None
    scale: float = 1.0

    def __call__(self, values: np.ndarray) -> float:
        return self.value(values)


def _as_float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def list_objectives() -> list[str]:
    """Return objective names supported by the benchmark registry."""

    return [
        "rosenbrock",
        "rastrigin",
        "himmelblau",
        "ackley",
        "beale",
        "booth",
        "styblinski_tang",
        "desmos",
        "desmos_surface",
    ]


def get_objective(name: str, dimension: int = 2, scale: float = 1.0) -> Objective:
    """Create an objective descriptor by registry name."""

    normalized = name.lower().replace("-", "_")
    if normalized == "rosenbrock":
        return Objective(
            name="rosenbrock",
            dimension=dimension,
            value=lambda values: float(RosenbrockValue(_as_float_array(values))),
            gradient=lambda values: RosenbrockGradient(_as_float_array(values)),
            hessian=lambda values: RosenbrockHessian(_as_float_array(values)),
            minimum=np.ones(dimension, dtype=np.float64),
            native_name="rosenbrock",
        )
    if normalized in {"desmos", "desmos_surface"}:
        return Objective(
            name="desmos_surface",
            dimension=2,
            value=lambda values: float(DesmosSurfaceValue(_as_float_array(values), scale=scale)),
            gradient=None,
            hessian=None,
            minimum=None,
            derivative_free=True,
            native_name="desmos_surface",
            scale=scale,
        )
    if normalized in {
        "rastrigin",
        "himmelblau",
        "ackley",
        "beale",
        "booth",
        "styblinski_tang",
    }:
        info = BenchmarkFunctionInfo(normalized, dimension)
        native_name = str(info["name"])
        fixed_dimension = int(info["fixed_dimension"])
        actual_dimension = fixed_dimension if fixed_dimension else dimension
        minimum = np.asarray(info["known_minimum"], dtype=np.float64)
        if minimum.size == 0:
            minimum = None
        return Objective(
            name=native_name,
            dimension=actual_dimension,
            value=lambda values: float(
                BenchmarkFunctionValue(native_name, _as_float_array(values), scale=scale)
            ),
            gradient=lambda values: (
                BenchmarkFunctionGradient(native_name, _as_float_array(values))
                if info["has_gradient"]
                else None
            ),
            hessian=lambda values: (
                BenchmarkFunctionHessian(native_name, _as_float_array(values))
                if info["has_hessian"]
                else None
            ),
            minimum=minimum,
            minimum_value=float(info["known_minimum_value"]),
            derivative_free=not bool(info["has_gradient"]),
            native_name=native_name,
            scale=scale,
        )
    msg = f"Unknown objective: {name}"
    raise ValueError(msg)


def desmos_numerical_gradient(
    values: np.ndarray,
    *,
    scale: float = 1.0,
    scheme: str = "central",
) -> np.ndarray:
    """Explicit numerical gradient helper for the nonsmooth Desmos surface."""

    return DesmosSurfaceNumericalGradient(_as_float_array(values), scale=scale, scheme=scheme)
