from __future__ import annotations

from pathlib import Path

import optlib

ROOT = Path(__file__).resolve().parents[2]
FIRST_DATASET = ROOT / "data" / "first_dataset.csv"


def test_compare_nn_optimizers_and_regularization_ablation() -> None:
    rows = optlib.compare_nn_optimizers(
        FIRST_DATASET,
        methods=["adam", "heavy_ball"],
        hidden_dim=8,
        learning_rate=0.03,
        max_iter=1200,
        seed=7,
        schedule="cosine",
    )

    assert [row["method"] for row in rows] == ["adam", "heavy_ball"]
    assert max(row["test_f1"] for row in rows) > 0.75
    assert all(row["schedule"] == "cosine" for row in rows)

    ablation = optlib.regularization_ablation(
        FIRST_DATASET,
        l2_values=[0.0, 1e-4],
        hidden_dim=8,
        max_iter=1000,
        seed=7,
    )
    assert [row["l2"] for row in ablation] == [0.0, 1e-4]
    assert all(row["test_f1"] > 0.7 for row in ablation)

    init_rows = optlib.initialization_ablation(
        FIRST_DATASET,
        initializations=["xavier", "he"],
        hidden_dim=8,
        max_iter=800,
        seed=7,
    )
    assert [row["initialization"] for row in init_rows] == ["xavier", "he"]
    assert all(0.0 <= row["test_f1"] <= 1.0 for row in init_rows)


def test_stability_and_weighted_score_helpers() -> None:
    stability = optlib.optimizer_stability(
        FIRST_DATASET,
        methods=["adam"],
        seeds=[1, 2],
        hidden_dim=6,
        max_iter=400,
    )
    assert [row["seed"] for row in stability] == [1, 2]
    assert all(row["method"] == "adam" for row in stability)

    score = optlib.weighted_f1_score(
        {"d1": FIRST_DATASET, "d2": None, "d3": None},
        hidden_dim=6,
        max_iter=400,
        seed=3,
    )
    assert score["missing"] == ["d2", "d3"]
    assert score["rows"][0]["dataset"] == "d1"
    assert 0.0 <= score["total"] <= 0.3


def test_sklearn_baseline_is_optional() -> None:
    baseline = optlib.sklearn_mlp_baseline(FIRST_DATASET, hidden_dim=6, max_iter=50, seed=3)
    if baseline is not None:
        assert baseline["method"] == "sklearn:MLPClassifier"
        assert 0.0 <= baseline["test_f1"] <= 1.0

    torch_baseline = optlib.torch_mlp_baseline(
        FIRST_DATASET,
        hidden_dim=6,
        epochs=5,
        seed=3,
    )
    if torch_baseline is not None:
        assert torch_baseline["method"] == "torch:Adam"
        assert 0.0 <= torch_baseline["test_f1"] <= 1.0
