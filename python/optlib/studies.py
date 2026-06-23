"""Neural-network study helpers for Lab 4."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import (
    binary_metrics,
    load_csv,
    load_dataset,
    standardize_train_test,
    stratified_split,
    train_binary_dataset,
)
from .nn import MLPClassifier

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


def initialization_ablation(
    path: str | Path,
    *,
    initializations: Iterable[str] = ("xavier", "he"),
    hidden_dim: int = 12,
    method: str = "adam",
    learning_rate: float = 0.03,
    max_iter: int = 3000,
    seed: int = 42,
    l2: float = 1e-4,
) -> list[dict[str, Any]]:
    """Compare supported parameter initializations for the native binary MLP."""

    rows: list[dict[str, Any]] = []
    for initialization in initializations:
        result = train_binary_dataset(
            path,
            hidden_dim=hidden_dim,
            method=method,
            learning_rate=learning_rate,
            max_iter=max_iter,
            seed=seed,
            initialization=initialization,
            l2=l2,
        )
        rows.append(
            {
                "initialization": initialization,
                "test_f1": result.test_metrics["f1"],
                "test_accuracy": result.test_metrics["accuracy"],
                "loss": result.model.classifier.loss_,
            }
        )
    return rows


def optimizer_stability(
    path: str | Path,
    *,
    methods: Iterable[str] = ("adam", "heavy_ball", "nesterov"),
    seeds: Iterable[int] = (1, 2, 3, 4, 5),
    hidden_dim: int = 12,
    learning_rate: float = 0.03,
    max_iter: int = 3000,
    schedule: str = "constant",
    l2: float = 1e-4,
) -> list[dict[str, Any]]:
    """Run optimizer comparisons across seeds and return one row per run."""

    rows: list[dict[str, Any]] = []
    for method in methods:
        for seed in seeds:
            result = train_binary_dataset(
                path,
                hidden_dim=hidden_dim,
                method=method,
                learning_rate=learning_rate,
                max_iter=max_iter,
                seed=seed,
                schedule=schedule,
                schedule_iterations=max_iter,
                l2=l2,
            )
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "test_f1": result.test_metrics["f1"],
                    "test_accuracy": result.test_metrics["accuracy"],
                    "loss": result.model.classifier.loss_,
                }
            )
    return rows


def _macro_f1_score(
    targets: np.ndarray,
    predictions: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    target_array = np.asarray(targets)
    predicted_array = np.asarray(predictions)
    labels = np.asarray(classes)
    matrix = np.zeros((labels.size, labels.size), dtype=np.int64)
    for actual_index, actual in enumerate(labels):
        actual_mask = target_array == actual
        for predicted_index, predicted in enumerate(labels):
            matrix[actual_index, predicted_index] = int(
                np.sum(actual_mask & (predicted_array == predicted))
            )

    f1_values: list[float] = []
    for index, label in enumerate(labels):
        true_positive = matrix[index, index]
        false_positive = int(np.sum(matrix[:, index]) - true_positive)
        false_negative = int(np.sum(matrix[index, :]) - true_positive)
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        denominator = precision + recall
        f1_values.append(2.0 * precision * recall / denominator if denominator else 0.0)

    return {
        "accuracy": float(np.mean(predicted_array == target_array)),
        "f1": float(np.mean(f1_values)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class_f1": dict(zip(labels.tolist(), f1_values, strict=True)),
        "confusion_matrix": matrix.tolist(),
    }


def train_dataset_score(
    path: str | Path,
    *,
    hidden_dim: int = 12,
    method: str = "adam",
    learning_rate: float = 0.03,
    max_iter: int = 3000,
    seed: int = 42,
    activation: str = "tanh",
    initialization: str = "xavier",
    l2: float = 1e-4,
    schedule: str = "constant",
) -> dict[str, Any]:
    """Train a binary or one-vs-rest MLP and return test F1 metadata."""

    features, targets = load_csv(path)
    classes = np.unique(targets)
    if classes.size <= 2:
        result = train_binary_dataset(
            path,
            hidden_dim=hidden_dim,
            method=method,
            learning_rate=learning_rate,
            max_iter=max_iter,
            seed=seed,
            activation=activation,
            initialization=initialization,
            l2=l2,
            schedule=schedule,
            schedule_iterations=max_iter,
        )
        return {
            "path": str(path),
            "mode": "binary",
            "class_count": int(classes.size),
            "test_f1": result.test_metrics["f1"],
            "test_accuracy": result.test_metrics["accuracy"],
            "loss": result.model.classifier.loss_,
        }

    x_train_raw, y_train, x_test_raw, y_test = stratified_split(features, targets, seed=seed)
    x_train, x_test, _, _ = standardize_train_test(x_train_raw, x_test_raw)
    probabilities: list[np.ndarray] = []
    losses: list[float] = []
    for label in classes:
        binary_targets = (y_train == label).astype(np.float64)
        classifier = MLPClassifier(
            hidden_dim=hidden_dim,
            method=method,
            learning_rate=learning_rate,
            max_iter=max_iter,
            seed=seed,
            activation=activation,
            initialization=initialization,
            l2=l2,
            schedule=schedule,
            schedule_iterations=max_iter,
        ).fit(x_train, binary_targets)
        probabilities.append(classifier.predict_proba(x_test))
        losses.append(float(classifier.loss_))
    score_matrix = np.column_stack(probabilities)
    predictions = classes[np.argmax(score_matrix, axis=1)]
    metrics = _macro_f1_score(y_test, predictions, classes)
    return {
        "path": str(path),
        "mode": "one_vs_rest",
        "class_count": int(classes.size),
        "test_f1": metrics["f1"],
        "test_accuracy": metrics["accuracy"],
        "loss": float(np.mean(losses)),
        "metrics": metrics,
    }


def weighted_f1_score(
    paths: dict[str, str | Path | None],
    *,
    weights: dict[str, float] | None = None,
    hidden_dim: int = 12,
    method: str = "adam",
    learning_rate: float = 0.03,
    max_iter: int = 3000,
    seed: int = 42,
    activation: str = "tanh",
    initialization: str = "xavier",
    l2: float = 1e-4,
    schedule: str = "constant",
) -> dict[str, Any]:
    """Compute weighted F1 over available datasets with binary/OVR fallback."""

    score_weights = {"d1": 0.3, "d2": 0.3, "d3": 0.4} if weights is None else weights
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    total = 0.0
    for name, weight in score_weights.items():
        raw_path = paths.get(name)
        if raw_path is None or not Path(raw_path).exists():
            missing.append(name)
            continue
        result = train_dataset_score(
            raw_path,
            hidden_dim=hidden_dim,
            method=method,
            learning_rate=learning_rate,
            max_iter=max_iter,
            seed=seed,
            activation=activation,
            initialization=initialization,
            l2=l2,
            schedule=schedule,
        )
        contribution = weight * result["test_f1"]
        rows.append({"dataset": name, "weight": weight, "contribution": contribution, **result})
        total += contribution
    return {"rows": rows, "total": total, "missing": missing, "weights": score_weights}


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


def torch_mlp_baseline(
    path: str | Path,
    *,
    hidden_dim: int = 12,
    learning_rate: float = 0.03,
    epochs: int = 500,
    seed: int = 42,
) -> dict[str, Any] | None:
    """Run a compact CPU PyTorch baseline when torch is installed."""

    try:
        import torch
    except ModuleNotFoundError:
        return None

    split = load_dataset(path, seed=seed)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(split.x_train.shape[1], hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dim, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    x_train = torch.as_tensor(split.x_train, dtype=torch.float32)
    y_train = torch.as_tensor(split.y_train.reshape(-1, 1), dtype=torch.float32)
    x_test = torch.as_tensor(split.x_test, dtype=torch.float32)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x_test)).numpy().ravel()
    metrics = binary_metrics(split.y_test, probabilities)
    return {
        "method": "torch:Adam",
        "hidden_dim": hidden_dim,
        "test_f1": metrics["f1"],
        "test_accuracy": metrics["accuracy"],
        "loss": float(loss.item()),
        "epochs": epochs,
    }
