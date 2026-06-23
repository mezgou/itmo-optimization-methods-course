from __future__ import annotations

import math

import optlib


def test_learning_rate_step_schedule() -> None:
    assert (
        optlib.LearningRateAt(0, "step", initial_learning_rate=0.1, gamma=0.5, step_size=3) == 0.1
    )
    assert (
        optlib.LearningRateAt(3, "step", initial_learning_rate=0.1, gamma=0.5, step_size=3) == 0.05
    )


def test_learning_rate_exponential_schedule() -> None:
    value = optlib.LearningRateAt(
        5,
        "exponential",
        initial_learning_rate=0.2,
        decay_rate=0.1,
    )
    assert math.isclose(value, 0.2 * math.exp(-0.5))


def test_learning_rate_cosine_warmup() -> None:
    value = optlib.LearningRateAt(
        0,
        "cosine",
        initial_learning_rate=0.1,
        minimum_learning_rate=0.01,
        total_iterations=10,
        warmup_steps=2,
    )
    assert math.isclose(value, 0.05)
