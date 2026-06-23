"""Small Python wrapper around the native binary MLP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from ._optlib import (
    BinaryF1Score,
    BinaryMlpLossAndGradient,
    BinaryMlpParameterCount,
    BinaryMlpPredictProba,
    TrainBinaryMlp,
)


@dataclass
class MLPClassifier:
    """Binary MLP classifier backed by the C++ core."""

    hidden_dim: int = 8
    activation: str = "tanh"
    method: str = "adam"
    learning_rate: float = 0.01
    max_iter: int = 5000
    gradient_tolerance: float = 1e-5
    initialization: str = "xavier"
    seed: int = 42
    l2: float = 0.0
    schedule: str = "constant"
    learning_rate_gamma: float = 0.5
    learning_rate_step_size: int = 100
    learning_rate_decay: float = 1e-3
    minimum_learning_rate: float = 0.0
    warmup_steps: int = 0
    schedule_iterations: int = 0
    log_trajectory: bool = False

    parameters_: np.ndarray | None = None
    input_dim_: int | None = None
    loss_: float | None = None
    f1_: float | None = None
    optimizer_result_: dict[str, Any] | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> Self:
        feature_array = np.asarray(features, dtype=np.float64)
        target_array = np.asarray(targets, dtype=np.float64)
        result = TrainBinaryMlp(
            feature_array,
            target_array,
            hidden_dim=self.hidden_dim,
            method=self.method,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            gradient_tolerance=self.gradient_tolerance,
            activation=self.activation,
            initialization=self.initialization,
            seed=self.seed,
            l2=self.l2,
            schedule=self.schedule,
            learning_rate_gamma=self.learning_rate_gamma,
            learning_rate_step_size=self.learning_rate_step_size,
            learning_rate_decay=self.learning_rate_decay,
            minimum_learning_rate=self.minimum_learning_rate,
            warmup_steps=self.warmup_steps,
            schedule_iterations=self.schedule_iterations,
            log_trajectory=self.log_trajectory,
        )
        self.parameters_ = np.asarray(result["parameters"], dtype=np.float64)
        self.input_dim_ = feature_array.shape[1]
        self.loss_ = float(result["loss"])
        self.f1_ = float(result["f1"])
        self.optimizer_result_ = result.get("optimizer_result")
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return BinaryMlpPredictProba(
            self.parameters_,
            np.asarray(features, dtype=np.float64),
            self.input_dim_,
            self.hidden_dim,
            self.activation,
        )

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(np.int64)

    def score(self, features: np.ndarray, targets: np.ndarray) -> float:
        probabilities = self.predict_proba(features)
        return float(BinaryF1Score(probabilities, np.asarray(targets, dtype=np.float64)))

    def get_parameters(self) -> np.ndarray:
        self._require_fitted()
        return np.array(self.parameters_, copy=True)

    def set_parameters(self, parameters: np.ndarray, input_dim: int) -> Self:
        parameter_array = np.asarray(parameters, dtype=np.float64)
        expected = BinaryMlpParameterCount(input_dim, self.hidden_dim)
        if parameter_array.shape != (expected,):
            msg = f"expected parameter vector of shape ({expected},), got {parameter_array.shape}"
            raise ValueError(msg)
        self.parameters_ = np.array(parameter_array, copy=True)
        self.input_dim_ = input_dim
        return self

    def _require_fitted(self) -> None:
        if self.parameters_ is None or self.input_dim_ is None:
            msg = "MLPClassifier is not fitted"
            raise RuntimeError(msg)


def make_xor() -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic XOR toy dataset."""

    return (
        np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
    )


def make_two_moons(
    samples: int = 200, noise: float = 0.08, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic two-moons dataset without sklearn."""

    if samples < 2:
        raise ValueError("samples must be at least 2")
    rng = np.random.default_rng(seed)
    half = samples // 2
    angles_a = rng.uniform(0.0, np.pi, size=half)
    angles_b = rng.uniform(0.0, np.pi, size=samples - half)
    first = np.column_stack([np.cos(angles_a), np.sin(angles_a)])
    second = np.column_stack([1.0 - np.cos(angles_b), 0.5 - np.sin(angles_b)])
    features = np.vstack([first, second])
    targets = np.concatenate([np.zeros(half), np.ones(samples - half)])
    features += rng.normal(0.0, noise, size=features.shape)
    return features.astype(np.float64), targets.astype(np.float64)


def binary_mlp_loss_and_gradient(
    parameters: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    hidden_dim: int,
    activation: str = "tanh",
    l2: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Expose native BCE loss and flat backprop gradient for checks."""

    feature_array = np.asarray(features, dtype=np.float64)
    return BinaryMlpLossAndGradient(
        np.asarray(parameters, dtype=np.float64),
        feature_array,
        np.asarray(targets, dtype=np.float64),
        feature_array.shape[1],
        hidden_dim,
        activation,
        l2,
    )
