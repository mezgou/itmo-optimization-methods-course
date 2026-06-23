"""Python entry point for the optlib C++ extension."""

from __future__ import annotations

from ._optlib import (
    Add,
    Axpy,
    DifferentiationScheme,
    Dot,
    Gemm,
    Gemv,
    Norm2,
    NormInf,
    NumericGradient,
    RosenbrockAutogradGradient,
    RosenbrockGradient,
    RosenbrockGradientAutograd,
    RosenbrockGradientNumeric,
    RosenbrockHessian,
    RosenbrockNumericalGradient,
    RosenbrockValue,
    Version,
)

__version__ = Version()
__all__ = [
    "Add",
    "Axpy",
    "DifferentiationScheme",
    "Dot",
    "Gemm",
    "Gemv",
    "Norm2",
    "NormInf",
    "NumericGradient",
    "RosenbrockAutogradGradient",
    "RosenbrockGradient",
    "RosenbrockGradientAutograd",
    "RosenbrockGradientNumeric",
    "RosenbrockHessian",
    "RosenbrockNumericalGradient",
    "RosenbrockValue",
    "Version",
    "__version__",
]
