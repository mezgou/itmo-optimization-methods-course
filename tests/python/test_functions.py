from __future__ import annotations

import numpy as np

import optlib


def test_rastrigin_himmelblau_ackley_minima() -> None:
    np.testing.assert_allclose(optlib.RastriginValue(np.zeros(5)), 0.0, atol=1e-12)
    np.testing.assert_allclose(optlib.RastriginGradient(np.zeros(3)), np.zeros(3), atol=1e-12)

    himmelblau_minimum = np.array([3.0, 2.0])
    np.testing.assert_allclose(optlib.HimmelblauValue(himmelblau_minimum), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        optlib.HimmelblauGradient(himmelblau_minimum), np.zeros(2), atol=1e-12
    )

    np.testing.assert_allclose(optlib.AckleyValue(np.zeros(4)), 0.0, atol=1e-12)
    np.testing.assert_allclose(optlib.AckleyGradient(np.zeros(4)), np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(optlib.BealeValue(np.array([3.0, 0.5])), 0.0, atol=1e-12)
    np.testing.assert_allclose(optlib.BoothValue(np.array([1.0, 3.0])), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        optlib.StyblinskiTangValue(np.array([-2.903534])), -39.16616570377142, atol=1e-5
    )


def test_gradients_match_autograd_and_numeric() -> None:
    point = np.array([0.4, -0.7, 1.2])
    np.testing.assert_allclose(
        optlib.RastriginGradient(point),
        optlib.RastriginAutogradGradient(point),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        optlib.RastriginGradient(point),
        optlib.NumericGradient(optlib.RastriginValue, point, scheme="five_point"),
        rtol=1e-6,
        atol=1e-6,
    )

    himmelblau_point = np.array([2.5, -1.1])
    np.testing.assert_allclose(
        optlib.HimmelblauGradient(himmelblau_point),
        optlib.HimmelblauAutogradGradient(himmelblau_point),
        rtol=1e-10,
        atol=1e-10,
    )


def test_desmos_surface_formula_and_black_box_gradient_shape() -> None:
    point = np.array([1.0, 2.0])
    y_stripe = np.round(np.sin(10.0 * point[1])) + 2.0
    x_stripe = np.round(np.sin(7.0 * point[0])) + 2.0
    first = (point[0] * y_stripe) ** 2 + point[1] - 10.0
    second = point[0] + (point[1] * x_stripe) ** 2 - 7.0
    expected = first * first + second * second

    assert np.isclose(optlib.DesmosSurfaceValue(point), expected)
    assert optlib.DesmosSurfaceNumericalGradient(point).shape == (2,)


def test_objective_registry_and_experiment_harness() -> None:
    assert {
        "rosenbrock",
        "rastrigin",
        "himmelblau",
        "ackley",
        "beale",
        "booth",
        "styblinski_tang",
        "desmos_surface",
    }.issubset(set(optlib.list_objectives()))

    objective = optlib.get_objective("rastrigin", dimension=2)
    assert objective.dimension == 2
    np.testing.assert_allclose(objective.minimum, np.zeros(2))
    assert objective.value(np.zeros(2)) == 0.0

    rows = optlib.compare_methods(
        objective,
        np.array([0.3, -0.2]),
        methods=["lbfgs", "nelder_mead"],
        max_iter=200,
        gradient_tolerance=1e-6,
        step_tolerance=1e-8,
        function_tolerance=1e-10,
        log_trajectory=False,
    )
    assert [row["method"] for row in rows] == ["lbfgs", "nelder_mead"]
    assert all(row["function_evaluations"] > 0 for row in rows)
