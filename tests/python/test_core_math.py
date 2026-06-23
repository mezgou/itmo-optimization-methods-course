from __future__ import annotations

import numpy as np

import optlib


def test_linalg_vector_operations() -> None:
    left = np.array([1.0, 2.0, 3.0])
    right = np.array([4.0, -5.0, 6.0])

    assert optlib.Dot(left, right) == 12.0
    assert np.isclose(optlib.Norm2(left), np.sqrt(14.0))
    assert optlib.NormInf(right) == 6.0
    np.testing.assert_allclose(optlib.Axpy(2.0, left, right), np.array([6.0, -1.0, 12.0]))


def test_linalg_matrix_operations() -> None:
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    vector = np.array([1.0, 0.5, -1.0])
    right = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]])

    np.testing.assert_allclose(optlib.Gemv(matrix, vector), np.array([-1.0, 0.5]))
    np.testing.assert_allclose(optlib.Gemm(matrix, right), np.array([[4.0, 11.5], [13.0, 31.0]]))


def test_rosenbrock_value_gradient_and_hessian() -> None:
    point = np.array([-1.2, 1.0])

    assert np.isclose(optlib.RosenbrockValue(point), 24.2)
    np.testing.assert_allclose(optlib.RosenbrockGradient(point), np.array([-215.6, -88.0]))
    np.testing.assert_allclose(
        optlib.RosenbrockHessian(point),
        np.array([[1330.0, 480.0], [480.0, 200.0]]),
    )


def test_numeric_and_autograd_gradients_match_analytic() -> None:
    point = np.array([-1.2, 1.0, 0.7, 1.3, -0.4])
    analytic = optlib.RosenbrockGradient(point)

    for scheme, tolerance in [
        ("forward", 5e-4),
        ("central", 1e-5),
        ("five_point", 1e-7),
    ]:
        np.testing.assert_allclose(
            optlib.RosenbrockNumericalGradient(point, scheme=scheme),
            analytic,
            rtol=tolerance,
            atol=tolerance,
        )

    np.testing.assert_allclose(
        optlib.RosenbrockAutogradGradient(point),
        analytic,
        rtol=1e-10,
        atol=1e-10,
    )


def test_numeric_gradient_accepts_python_callable() -> None:
    point = np.array([1.5, -2.0, 0.5])

    def objective(values: np.ndarray) -> float:
        return float(np.sum(values * values))

    np.testing.assert_allclose(
        optlib.NumericGradient(objective, point, scheme="central"),
        2.0 * point,
        rtol=1e-6,
        atol=1e-6,
    )
