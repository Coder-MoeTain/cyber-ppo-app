import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_cc4_results import (
    two_level_bootstrap_difference, model_vs_heuristic_bootstrap,
)


def test_two_level_bootstrap_detects_known_positive_difference():
    a = np.full((4, 20), 10.0)
    b = np.full((4, 20), 7.0)
    result = two_level_bootstrap_difference(a, b, n_boot=500, seed=7)
    assert result["mean_difference"] == 3.0
    assert result["ci_95"][0] <= 3.0 <= result["ci_95"][1]


def test_model_heuristic_bootstrap_uses_shared_eval_seeds():
    model = np.full((3, 12), 5.0)
    heuristic = np.full(12, 2.0)
    result = model_vs_heuristic_bootstrap(model, heuristic, n_boot=500, seed=9)
    assert result["mean_difference"] == 3.0
    assert result["n_training_seeds"] == 3
    assert result["n_evaluation_seeds"] == 12
