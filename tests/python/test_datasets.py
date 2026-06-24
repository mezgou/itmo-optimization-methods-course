from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import optlib

ROOT = Path(__file__).resolve().parents[2]
FIRST_DATASET = ROOT / "data" / "first_dataset.csv"
SECOND_DATASET = ROOT / "data" / "second_dataset.csv"


def test_load_csv_split_and_standardize_first_dataset() -> None:
    features, targets = optlib.load_csv(FIRST_DATASET)
    assert features.shape == (600, 2)
    assert targets.shape == (600,)

    x_train, y_train, x_test, y_test = optlib.stratified_split(features, targets, seed=123)
    assert x_train.shape == (480, 2)
    assert x_test.shape == (120, 2)
    assert np.bincount(y_train.astype(np.int64)).tolist() == [240, 240]
    assert np.bincount(y_test.astype(np.int64)).tolist() == [60, 60]

    standardizer = optlib.fit_standardizer(x_train)
    assert np.allclose(np.mean(standardizer.transform(x_train), axis=0), 0.0, atol=1e-12)

    split = optlib.prepare_dataset(FIRST_DATASET, seed=123)
    assert split.train_features.shape == (480, 2)
    assert split.test_features.shape == (120, 2)
    assert np.allclose(np.mean(split.train_features, axis=0), 0.0, atol=1e-12)


def test_classification_metrics_return_binary_confusion_matrix() -> None:
    metrics = optlib.classification_metrics(
        np.array([0, 1, 1, 0]),
        np.array([0, 1, 0, 0]),
    )
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] > 0.66
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_download_uses_gdown(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, str] = {}

    def fake_download(*, id: str, output: str, quiet: bool) -> str:
        calls["id"] = id
        calls["output"] = output
        calls["quiet"] = str(quiet)
        Path(output).write_text("feature_0,target\n0,0\n", encoding="utf-8")
        return output

    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=fake_download))
    destination = tmp_path / "nested" / "dataset.csv"

    result = optlib.download("file-id", destination)

    assert result == destination
    assert destination.exists()
    assert calls == {"id": "file-id", "output": str(destination), "quiet": "False"}


def test_train_binary_classifier_scores_first_dataset() -> None:
    model, split, metrics = optlib.train_binary_classifier(
        FIRST_DATASET,
        method="adam",
        hidden_dim=8,
        learning_rate=0.03,
        max_iter=2000,
        seed=9,
    )
    assert metrics["f1"] > 0.8
    evaluated = optlib.evaluate(model, FIRST_DATASET, split.standardizer)
    assert evaluated["f1"] > 0.8


def test_binary_dataset_model_save_load(tmp_path: Path) -> None:
    model, _, metrics = optlib.train_binary_classifier(
        FIRST_DATASET,
        method="adam",
        hidden_dim=8,
        learning_rate=0.03,
        max_iter=2000,
        seed=9,
    )
    assert metrics["f1"] > 0.8

    path = model.save(tmp_path / "model.npz")
    loaded = optlib.load_binary_dataset_model(path)
    evaluated = loaded.evaluate_path(FIRST_DATASET)
    evaluated_from_path = optlib.evaluate_saved_model(path, FIRST_DATASET)

    assert evaluated["f1"] > 0.8
    assert evaluated_from_path["f1"] > 0.8

    suffixless_path = model.save(tmp_path / "model_without_suffix")
    assert suffixless_path.name == "model_without_suffix.npz"
    assert suffixless_path.exists()


def test_train_binary_classifier_scores_second_dataset() -> None:
    _, _, metrics = optlib.train_binary_classifier(
        SECOND_DATASET,
        method="adam",
        hidden_dim=12,
        learning_rate=0.03,
        max_iter=2500,
        seed=10,
    )
    assert metrics["f1"] > 0.8
