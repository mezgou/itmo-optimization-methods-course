"""Neural-network study helpers for Lab 4."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import binary_metrics, load_dataset, train_binary_dataset

DEFAULT_NN_METHODS = ["gradient_descent", "heavy_ball", "nesterov", "adam", "rmsprop", "adagrad"]


def compare_nn_optimizers(
    path: str | Path,
    *,
    methods: Iterable[str] = DEFAULT_NN_METHODS,
    hidden_dim: int = 12,
    learning_rate: float = 0.03,
    max_iter: int = 3000,
    seed: int = 42,
    schedule: str = "constant",
) -> list[dict[str, Any]]:
    """Train the same binary MLP with several optlib first-order optimizers."""

    rows: list[dict[str, Any]] = []
    for method in methods:
        result = train_binary_dataset(
            path,
            hidden_dim=hidden_dim,
            method=method,
            learning_rate=learning_rate,
            max_iter=max_iter,
            seed=seed,
            schedule=schedule,
            schedule_iterations=max_iter,
        )
        rows.append(
            {
                "method": method,
                "schedule": schedule,
                "hidden_dim": hidden_dim,
                "l2": result.model.classifier.l2,
                "train_f1": result.train_metrics["f1"],
                "test_f1": result.test_metrics["f1"],
                "test_accuracy": result.test_metrics["accuracy"],
                "loss": result.model.classifier.loss_,
            }
        )
    return rows


def regularization_ablation(
    path: str | Path,
    *,
    l2_values: Iterable[float] = (0.0, 1e-4, 1e-3),
    hidden_dim: int = 12,
    method: str = "adam",
    learning_rate: float = 0.03,
    max_iter: int = 3000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Compare L2 values for the native binary MLP."""

    rows: list[dict[str, Any]] = []
    for l2 in l2_values:
        split = load_dataset(path, seed=seed)
        from .nn import MLPClassifier

        classifier = MLPClassifier(
            hidden_dim=hidden_dim,
            method=method,
            learning_rate=learning_rate,
            max_iter=max_iter,
            seed=seed,
            l2=l2,
        ).fit(split.x_train, split.y_train)
        probabilities = classifier.predict_proba(split.x_test)
        metrics = binary_metrics(split.y_test, probabilities)
        rows.append({"l2": l2, "test_f1": metrics["f1"], "loss": classifier.loss_})
    return rows


def sklearn_mlp_baseline(
    path: str | Path,
    *,
    hidden_dim: int = 12,
    max_iter: int = 1000,
    seed: int = 42,
) -> dict[str, Any] | None:
    """Run sklearn MLPClassifier when scikit-learn is installed."""

    try:
        from sklearn.neural_network import MLPClassifier as SklearnMlpClassifier
    except ModuleNotFoundError:
        return None

    split = load_dataset(path, seed=seed)
    classifier = SklearnMlpClassifier(
        hidden_layer_sizes=(hidden_dim,),
        activation="tanh",
        solver="adam",
        max_iter=max_iter,
        random_state=seed,
    )
    classifier.fit(split.x_train, split.y_train.astype(np.int64))
    probabilities = classifier.predict_proba(split.x_test)[:, 1]
    metrics = binary_metrics(split.y_test, probabilities)
    return {
        "method": "sklearn:MLPClassifier",
        "hidden_dim": hidden_dim,
        "test_f1": metrics["f1"],
        "test_accuracy": metrics["accuracy"],
        "iterations": int(classifier.n_iter_),
    }
