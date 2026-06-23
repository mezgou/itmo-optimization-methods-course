from __future__ import annotations

from pathlib import Path

import numpy as np

import optlib

ROOT = Path(__file__).resolve().parents[2]


def test_load_and_split_dataset() -> None:
    features, targets = optlib.load_csv(ROOT / "data" / "first_dataset.csv")
    assert features.shape == (600, 2)
    assert targets.shape == (600,)

    split = optlib.prepare_dataset(ROOT / "data" / "first_dataset.csv", seed=123)
    assert split.train_features.shape == (480, 2)
    assert split.test_features.shape == (120, 2)
    assert np.allclose(np.mean(split.train_features, axis=0), 0.0, atol=1e-12)


def test_classification_metrics() -> None:
    metrics = optlib.classification_metrics(
        np.array([0, 1, 1, 0]),
        np.array([0, 1, 0, 0]),
    )
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["f1"] > 0.66
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_train_binary_classifier_on_first_dataset() -> None:
    model, split, metrics = optlib.train_binary_classifier(
        ROOT / "data" / "first_dataset.csv",
        method="adam",
        hidden_dim=8,
        learning_rate=0.03,
        max_iter=2000,
        seed=9,
    )
    assert metrics["f1"] > 0.8
    evaluated = optlib.evaluate(model, ROOT / "data" / "first_dataset.csv", split.standardizer)
    assert evaluated["f1"] > 0.8


def test_train_binary_classifier_on_second_dataset() -> None:
    _, _, metrics = optlib.train_binary_classifier(
        ROOT / "data" / "second_dataset.csv",
        method="adam",
        hidden_dim=12,
        learning_rate=0.03,
        max_iter=2500,
        seed=10,
    )
    assert metrics["f1"] > 0.8
