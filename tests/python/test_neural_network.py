from __future__ import annotations

import numpy as np

import optlib


def test_binary_mlp_loss_gradient_shapes() -> None:
    features, targets = optlib.make_xor()
    parameters = optlib.InitializeBinaryMlpParameters(2, 4, seed=7)
    result = optlib.BinaryMlpLossAndGradient(parameters, features, targets, 2, 4)

    assert result["loss"] > 0.0
    assert result["gradient"].shape == parameters.shape
    assert optlib.BinaryMlpParameterCount(2, 4) == parameters.shape[0]


def test_mlp_classifier_fits_xor() -> None:
    features, targets = optlib.make_xor()
    classifier = optlib.MLPClassifier(
        hidden_dim=8,
        learning_rate=0.05,
        max_iter=5000,
        gradient_tolerance=1e-5,
        seed=11,
    ).fit(features, targets)

    assert classifier.loss_ is not None and classifier.loss_ < 0.05
    assert classifier.score(features, targets) > 0.99
    np.testing.assert_array_equal(classifier.predict(features), targets.astype(np.int64))


def test_mlp_classifier_two_moons_sanity() -> None:
    features, targets = optlib.make_two_moons(samples=120, noise=0.05, seed=4)
    classifier = optlib.MLPClassifier(
        hidden_dim=12,
        learning_rate=0.03,
        max_iter=4000,
        gradient_tolerance=1e-5,
        seed=5,
    ).fit(features, targets)

    assert classifier.score(features, targets) > 0.9
