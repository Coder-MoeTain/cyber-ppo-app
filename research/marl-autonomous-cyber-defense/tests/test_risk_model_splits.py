"""Tests for strict separation of risk-model fitting, calibration and final evaluation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).parents[1] / "scripts" / "train_risk_model.py"
SPEC = importlib.util.spec_from_file_location("train_risk_model", MODULE)
RISK_TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RISK_TRAIN)


def _dataset(path, seeds, labels=None):
    labels = np.asarray(labels if labels is not None else [0, 1, 0, 1], dtype=np.int64)
    episode_ids = np.repeat(np.asarray(seeds, dtype=np.int64), len(labels) // len(seeds))
    np.savez_compressed(
        path,
        observations=np.ones((len(labels), 3), dtype=np.float32),
        agent_ids=np.zeros(len(labels), dtype=np.int64),
        labels=labels,
        episode_ids=episode_ids,
        agent_names=np.asarray(["blue_agent_0"]),
        seeds=np.asarray(seeds, dtype=np.int64),
    )


def test_risk_splits_must_be_disjoint():
    with pytest.raises(ValueError, match="overlap"):
        RISK_TRAIN.assert_disjoint_splits([
            ("train", {1, 2}), ("validation", {3}),
            ("calibration", {2, 4}), ("test", {5}),
        ])


def test_risk_splits_accept_disjoint_episode_seeds():
    RISK_TRAIN.assert_disjoint_splits([
        ("train", {1, 2}), ("validation", {3}),
        ("calibration", {4}), ("test", {5}),
    ])


def test_temperature_is_fitted_only_from_binary_calibration_labels():
    logits = np.asarray([-2.0, -1.0, 1.0, 2.0])
    labels = np.asarray([0, 0, 1, 1])
    temperature = RISK_TRAIN.fit_temperature(logits, labels)
    assert 0.25 <= temperature <= 4.0
    with pytest.raises(ValueError, match="both binary classes"):
        RISK_TRAIN.fit_temperature(logits, np.zeros(4, dtype=int))


def test_probability_transform_is_finite_for_extreme_logits():
    probs = RISK_TRAIN.probabilities(np.asarray([-1e9, 0.0, 1e9]), 1.0)
    assert np.isfinite(probs).all()
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_dataset_loader_checks_seed_provenance_and_binary_labels(tmp_path):
    path = tmp_path / "risk.npz"
    _dataset(path, [10, 11])
    data, names, seeds = RISK_TRAIN.load_dataset(path)
    assert names == ["blue_agent_0"]
    assert seeds == {10, 11}
    assert len(data["labels"]) == 4

    invalid = tmp_path / "invalid.npz"
    _dataset(invalid, [10, 11], labels=[0, 1, 2, 1])
    with pytest.raises(ValueError, match="binary"):
        RISK_TRAIN.load_dataset(invalid)
