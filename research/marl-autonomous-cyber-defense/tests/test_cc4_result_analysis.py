"""Regression tests for the paired CC4 analysis data contract."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).parents[1] / "scripts" / "analyze_cc4_results.py"
SPEC = importlib.util.spec_from_file_location("analyze_cc4_results", MODULE)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def test_model_matrix_preserves_seed_pairing():
    rows = [
        {"training_seed": 11, "seed": 501, "team_reward_sum": 1.0},
        {"training_seed": 11, "seed": 502, "team_reward_sum": 2.0},
        {"training_seed": 22, "seed": 501, "team_reward_sum": 3.0},
        {"training_seed": 22, "seed": 502, "team_reward_sum": 4.0},
    ]
    train, evaluation, matrix = ANALYSIS.model_matrix(rows)
    assert train == [11, 22]
    assert evaluation == [501, 502]
    np.testing.assert_array_equal(matrix, [[1.0, 2.0], [3.0, 4.0]])


def test_model_matrix_rejects_duplicate_seed_pair():
    rows = [
        {"training_seed": 11, "seed": 501, "team_reward_sum": 1.0},
        {"training_seed": 11, "seed": 501, "team_reward_sum": 2.0},
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        ANALYSIS.model_matrix(rows)


def test_model_matrix_rejects_incomplete_grid():
    rows = [
        {"training_seed": 11, "seed": 501, "team_reward_sum": 1.0},
        {"training_seed": 22, "seed": 502, "team_reward_sum": 2.0},
    ]
    with pytest.raises(ValueError, match="incomplete"):
        ANALYSIS.model_matrix(rows)


def test_heuristic_vector_rejects_duplicate_evaluation_seed():
    rows = [
        {"seed": 501, "team_reward_sum": -1.0},
        {"seed": 501, "team_reward_sum": -2.0},
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        ANALYSIS.heuristic_vector(rows)


def test_two_level_bootstrap_is_reproducible_and_directional():
    a = np.array([[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]])
    b = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    first = ANALYSIS.two_level_bootstrap_difference(a, b, n_boot=500, seed=7)
    second = ANALYSIS.two_level_bootstrap_difference(a, b, n_boot=500, seed=7)
    assert first == second
    assert first["mean_difference"] == pytest.approx(3.0)
    assert first["ci_95"][0] > 0


def test_two_level_bootstrap_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        ANALYSIS.two_level_bootstrap_difference(np.ones(3), np.ones(3))


def test_expected_matrix_rejects_partial_training_seed_set():
    with pytest.raises(ValueError, match="Training seed set mismatch"):
        ANALYSIS.validate_expected_matrix(
            [11, 22], [501, 502],
            expected_training=[11, 22, 33],
            expected_evaluation=[501, 502],
        )


def test_expected_matrix_rejects_partial_held_out_seed_set():
    with pytest.raises(ValueError, match="Evaluation seed set mismatch"):
        ANALYSIS.validate_expected_matrix(
            [11, 22], [501, 502],
            expected_training=[11, 22],
            expected_evaluation=[501, 502, 503],
        )


def test_expected_matrix_accepts_exact_preregistered_seed_sets():
    ANALYSIS.validate_expected_matrix(
        [11, 22], [501, 502],
        expected_training=[22, 11],
        expected_evaluation=[502, 501],
    )
