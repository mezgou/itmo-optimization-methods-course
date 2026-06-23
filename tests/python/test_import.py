from __future__ import annotations

import numpy as np

import optlib


def test_import_and_version() -> None:
    assert optlib.__version__ == "1.0.0"
    assert optlib.Version() == "1.0.0"


def test_extension_smoke_function() -> None:
    assert optlib.Add(2.0, 3.0) == 5.0


def test_linalg_bindings() -> None:
    left_values = np.array([1.0, 2.0, 3.0])
    right_values = np.array([4.0, 5.0, 6.0])
    assert optlib.Dot(left_values, right_values) == 32.0
    assert np.isclose(optlib.Norm2(left_values), np.sqrt(14.0))
    assert optlib.NormInf(np.array([-1.0, 4.0, -3.0])) == 4.0


def test_rosenbrock_bindings() -> None:
    x_values = np.array([-1.2, 1.0, 0.8])
    analytic_gradient = optlib.RosenbrockGradient(x_values)
    autograd_gradient = optlib.RosenbrockAutogradGradient(x_values)
    central_gradient = optlib.RosenbrockNumericalGradient(x_values, "central")

    assert np.isclose(optlib.RosenbrockValue(np.ones(3)), 0.0)
    assert analytic_gradient.shape == (3,)
    assert np.allclose(autograd_gradient, analytic_gradient, atol=1e-10)
    assert np.allclose(central_gradient, analytic_gradient, atol=1e-4)

    hessian = optlib.RosenbrockHessian(np.ones(3))
    assert hessian.shape == (3, 3)
    assert np.isclose(hessian[0, 0], 802.0)


def test_numeric_gradient_python_callable() -> None:
    def objective(values: np.ndarray) -> float:
        return float(np.sum(values * values))

    gradient = optlib.NumericGradient(objective, np.array([1.0, -2.0, 3.0]), "five-point")
    assert np.allclose(gradient, np.array([2.0, -4.0, 6.0]), atol=1e-8)
