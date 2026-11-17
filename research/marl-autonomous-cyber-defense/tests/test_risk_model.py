import numpy as np
import torch
from autocd.risk_model import RiskEstimatorNet, RiskPredictor, binary_metrics


def test_risk_model_shapes_and_probabilities():
    model = RiskEstimatorNet(12, 5, 32)
    logits = model(torch.zeros(4, 12), torch.tensor([0, 1, 2, 4]))
    assert logits.shape == (4,)
    predictor = RiskPredictor(model, [f"blue_agent_{i}" for i in range(5)])
    score = predictor("blue_agent_0", np.zeros(12, dtype=np.float32), {})
    assert 0.0 <= score <= 1.0


def test_binary_metrics_known_perfect_ranking():
    result = binary_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0
    assert result["brier"] < 0.05


def test_risk_model_rejects_unknown_agent():
    model = RiskEstimatorNet(4, 1, 8)
    predictor = RiskPredictor(model, ["blue_agent_0"])
    try:
        predictor("blue_agent_1", np.zeros(4), {})
    except KeyError:
        return
    raise AssertionError("Unknown agent should be rejected")
