"""Dataset loading, preprocessing, and evaluation helpers for Labs 3-4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .nn import MLPClassifier

FIRST_DATASET_ID = "16VzwtrZRP9QScMdueb3BK0fvZtP8a6sH"
SECOND_DATASET_ID = "13npPTtGGiKyug7fEKyvaZZbFavFkhGLm"


@dataclass(frozen=True)
class DatasetSplit:
    """Stratified train/test split with train-fitted standardization."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    classes: np.ndarray


@dataclass(frozen=True)
class Standardizer:
    """Train-fitted feature standardizer."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=np.float64) - self.mean) / self.std


@dataclass(frozen=True)
class PreparedDataset:
    """Compatibility view used by Lab 3 tests and notebooks."""

    train_features: np.ndarray
    train_targets: np.ndarray
    test_features: np.ndarray
    test_targets: np.ndarray
    standardizer: Standardizer
    classes: np.ndarray


@dataclass
class BinaryDatasetModel:
    """Classifier plus preprocessing state used for d3 evaluation."""

    classifier: MLPClassifier
    mean: np.ndarray
    std: np.ndarray
    classes: np.ndarray | None = None

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=np.float64) - self.mean) / self.std

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.transform(features))

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(np.int64)

    def evaluate(self, features: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
        probabilities = self.predict_proba(features)
        return binary_metrics(targets, probabilities)

    def evaluate_path(self, path: str | Path) -> dict[str, Any]:
        features, targets = load_csv(path)
        return self.evaluate(features, targets)

    def save(self, path: str | Path) -> Path:
        """Persist model weights and preprocessing state for defense runs."""

        self.classifier._require_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            parameters=self.classifier.get_parameters(),
            input_dim=np.array([self.classifier.input_dim_], dtype=np.int64),
            hidden_dim=np.array([self.classifier.hidden_dim], dtype=np.int64),
            activation=np.array([self.classifier.activation]),
            method=np.array([self.classifier.method]),
            mean=self.mean,
            std=self.std,
            classes=np.array([] if self.classes is None else self.classes),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "BinaryDatasetModel":
        """Load a persisted binary dataset model."""

        with np.load(path, allow_pickle=False) as data:
            input_dim = int(data["input_dim"][0])
            hidden_dim = int(data["hidden_dim"][0])
            activation = str(data["activation"][0])
            method = str(data["method"][0])
            classifier = MLPClassifier(
                hidden_dim=hidden_dim,
                activation=activation,
                method=method,
            ).set_parameters(data["parameters"], input_dim)
            classes = np.array(data["classes"], copy=True)
            return cls(
                classifier=classifier,
                mean=np.array(data["mean"], copy=True),
                std=np.array(data["std"], copy=True),
                classes=None if classes.size == 0 else classes,
            )


@dataclass(frozen=True)
class DatasetTrainingResult:
    """Training result for one CSV dataset."""

    model: BinaryDatasetModel
    split: DatasetSplit
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]


def download(file_id: str, dest: str | Path) -> Path:
    """Download a Google Drive file by id using gdown."""

    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ModuleNotFoundError as error:
        msg = "gdown is required: run `uv sync --extra experiments --extra dev`"
        raise RuntimeError(msg) from error
    result = gdown.download(id=file_id, output=str(destination), quiet=False)
    if result is None:
        msg = f"gdown failed to download file id {file_id}"
        raise RuntimeError(msg)
    return destination


def load_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV whose last column is the target."""

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    names = data.dtype.names
    if names is None or len(names) < 2:
        msg = f"dataset {path} must contain feature columns and a target column"
        raise ValueError(msg)
    columns = [np.asarray(data[name], dtype=np.float64) for name in names]
    matrix = np.column_stack(columns)
    return matrix[:, :-1], matrix[:, -1]


def stratified_indices(
    targets: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic stratified train/test indices."""

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in np.unique(targets):
        indices = np.flatnonzero(targets == label)
        rng.shuffle(indices)
        test_count = max(1, int(round(indices.size * test_size)))
        test_parts.append(indices[:test_count])
        train_parts.append(indices[test_count:])
    train_indices = np.concatenate(train_parts)
    test_indices = np.concatenate(test_parts)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def stratified_split(
    features: np.ndarray,
    targets: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw train/test arrays using a deterministic stratified split."""

    feature_array = np.asarray(features, dtype=np.float64)
    target_array = np.asarray(targets)
    if feature_array.ndim != 2:
        raise ValueError("features must be a 2D array")
    if target_array.ndim != 1:
        raise ValueError("targets must be a 1D array")
    if feature_array.shape[0] != target_array.shape[0]:
        raise ValueError("features and targets must contain the same number of rows")
    train_indices, test_indices = stratified_indices(target_array, test_size=test_size, seed=seed)
    return (
        feature_array[train_indices],
        target_array[train_indices],
        feature_array[test_indices],
        target_array[test_indices],
    )


def fit_standardizer(features: np.ndarray) -> Standardizer:
    """Fit feature standardization parameters on a train matrix."""

    feature_array = np.asarray(features, dtype=np.float64)
    mean = np.mean(feature_array, axis=0)
    std = np.std(feature_array, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return Standardizer(mean=mean, std=std)


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit standardization on train and apply it to train/test."""

    standardizer = fit_standardizer(x_train)
    return (
        standardizer.transform(x_train),
        standardizer.transform(x_test),
        standardizer.mean,
        standardizer.std,
    )


def load_dataset(path: str | Path, test_size: float = 0.2, seed: int = 42) -> DatasetSplit:
    """Load, stratify, and standardize one CSV dataset."""

    features, targets = load_csv(path)
    x_train_raw, y_train, x_test_raw, y_test = stratified_split(
        features,
        targets,
        test_size=test_size,
        seed=seed,
    )
    x_train, x_test, mean, std = standardize_train_test(x_train_raw, x_test_raw)
    return DatasetSplit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        mean=mean,
        std=std,
        classes=np.unique(targets),
    )


def prepare_dataset(path: str | Path, test_size: float = 0.2, seed: int = 42) -> PreparedDataset:
    """Compatibility wrapper returning named train/test fields."""

    split = load_dataset(path, test_size=test_size, seed=seed)
    return PreparedDataset(
        train_features=split.x_train,
        train_targets=split.y_train,
        test_features=split.x_test,
        test_targets=split.y_test,
        standardizer=Standardizer(split.mean, split.std),
        classes=split.classes,
    )


def binary_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Compute binary accuracy, precision, recall, F1, and confusion matrix."""

    target_array = np.asarray(targets, dtype=np.float64)
    predicted = (np.asarray(probabilities, dtype=np.float64) >= threshold).astype(np.int64)
    actual = (target_array >= 0.5).astype(np.int64)
    true_positive = int(np.sum((predicted == 1) & (actual == 1)))
    true_negative = int(np.sum((predicted == 0) & (actual == 0)))
    false_positive = int(np.sum((predicted == 1) & (actual == 0)))
    false_negative = int(np.sum((predicted == 0) & (actual == 1)))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1 = 2.0 * precision * recall / f1_denominator if f1_denominator else 0.0
    accuracy = float(np.mean(predicted == actual))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
    }


def classification_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """Compatibility metrics for hard binary predictions."""

    metrics = binary_metrics(targets, predictions, threshold=0.5)
    metrics["confusion_matrix"] = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    return metrics


def train_binary_dataset(
    path: str | Path,
    *,
    hidden_dim: int = 12,
    method: str = "adam",
    learning_rate: float = 0.03,
    max_iter: int = 5000,
    gradient_tolerance: float = 1e-5,
    seed: int = 42,
    activation: str = "tanh",
    initialization: str = "xavier",
    l2: float = 0.0,
    schedule: str = "constant",
    learning_rate_gamma: float = 0.5,
    learning_rate_step_size: int = 100,
    learning_rate_decay: float = 1e-3,
    minimum_learning_rate: float = 0.0,
    warmup_steps: int = 0,
    schedule_iterations: int = 0,
    log_trajectory: bool = False,
) -> DatasetTrainingResult:
    """Train the native MLP on one binary CSV dataset."""

    split = load_dataset(path, seed=seed)
    classifier = MLPClassifier(
        hidden_dim=hidden_dim,
        method=method,
        learning_rate=learning_rate,
        max_iter=max_iter,
        gradient_tolerance=gradient_tolerance,
        activation=activation,
        initialization=initialization,
        seed=seed,
        l2=l2,
        schedule=schedule,
        learning_rate_gamma=learning_rate_gamma,
        learning_rate_step_size=learning_rate_step_size,
        learning_rate_decay=learning_rate_decay,
        minimum_learning_rate=minimum_learning_rate,
        warmup_steps=warmup_steps,
        schedule_iterations=schedule_iterations,
        log_trajectory=log_trajectory,
    ).fit(split.x_train, split.y_train)
    model = BinaryDatasetModel(
        classifier=classifier,
        mean=split.mean,
        std=split.std,
        classes=split.classes,
    )
    train_metrics = binary_metrics(split.y_train, classifier.predict_proba(split.x_train))
    test_metrics = binary_metrics(split.y_test, classifier.predict_proba(split.x_test))
    return DatasetTrainingResult(
        model=model,
        split=split,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )


def train_binary_classifier(
    path: str | Path,
    *,
    hidden_dim: int = 12,
    method: str = "adam",
    learning_rate: float = 0.03,
    max_iter: int = 5000,
    seed: int = 42,
    activation: str = "tanh",
    initialization: str = "xavier",
    l2: float = 0.0,
    schedule: str = "constant",
    learning_rate_gamma: float = 0.5,
    learning_rate_step_size: int = 100,
    learning_rate_decay: float = 1e-3,
    minimum_learning_rate: float = 0.0,
    warmup_steps: int = 0,
    schedule_iterations: int = 0,
    log_trajectory: bool = False,
) -> tuple[BinaryDatasetModel, PreparedDataset, dict[str, Any]]:
    """Compatibility wrapper returning model, prepared split, and test metrics."""

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
        learning_rate_gamma=learning_rate_gamma,
        learning_rate_step_size=learning_rate_step_size,
        learning_rate_decay=learning_rate_decay,
        minimum_learning_rate=minimum_learning_rate,
        warmup_steps=warmup_steps,
        schedule_iterations=schedule_iterations,
        log_trajectory=log_trajectory,
    )
    prepared = PreparedDataset(
        train_features=result.split.x_train,
        train_targets=result.split.y_train,
        test_features=result.split.x_test,
        test_targets=result.split.y_test,
        standardizer=Standardizer(result.split.mean, result.split.std),
        classes=result.split.classes,
    )
    return result.model, prepared, result.test_metrics


def evaluate(
    model: BinaryDatasetModel,
    path: str | Path,
    standardizer: Standardizer | None = None,
) -> dict[str, Any]:
    """Evaluate a trained dataset model on a CSV path, including closed d3."""

    if standardizer is None:
        return model.evaluate_path(path)
    features, targets = load_csv(path)
    probabilities = model.classifier.predict_proba(standardizer.transform(features))
    return binary_metrics(targets, probabilities)


def load_binary_dataset_model(path: str | Path) -> BinaryDatasetModel:
    """Load a model saved by BinaryDatasetModel.save."""

    return BinaryDatasetModel.load(path)


def run_lab3_experiment(
    paths: dict[str, str | Path] | list[str | Path] | tuple[str | Path, ...],
    *,
    optimizers: tuple[str, ...] = ("adam", "heavy_ball"),
    hidden_dim: int = 12,
    learning_rate: float = 0.03,
    max_iter: int = 5000,
    activation: str = "tanh",
    initialization: str = "xavier",
    l2: float = 0.0,
    schedule: str = "constant",
    log_trajectory: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Train d1/d2-style binary classifiers and return tabular experiment rows."""

    if isinstance(paths, dict):
        dataset_items = list(paths.items())
    else:
        dataset_items = [(Path(path).stem, path) for path in paths]

    rows: list[dict[str, Any]] = []
    for dataset_name, dataset_path in dataset_items:
        for method in optimizers:
            result = train_binary_dataset(
                dataset_path,
                hidden_dim=hidden_dim,
                method=method,
                learning_rate=learning_rate,
                max_iter=max_iter,
                activation=activation,
                initialization=initialization,
                l2=l2,
                schedule=schedule,
                log_trajectory=log_trajectory,
                seed=seed,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "train_f1": result.train_metrics["f1"],
                    "test_f1": result.test_metrics["f1"],
                    "accuracy": result.test_metrics["accuracy"],
                    "precision": result.test_metrics["precision"],
                    "recall": result.test_metrics["recall"],
                    "confusion_matrix": result.test_metrics["confusion_matrix"],
                    "loss": result.model.classifier.loss_,
                }
            )
    return rows
