import pytest
from autocd.risk import RiskScorer, RiskWeights

def test_risk_score():
    r = RiskScorer().assess(1, 1, 1, 1)
    assert r.score == pytest.approx(1.)
    assert r.severity == "high"

def test_invalid_value():
    with pytest.raises(ValueError): RiskScorer().assess(1.1, 0, 0, 0)

def test_bad_weights():
    with pytest.raises(ValueError): RiskScorer(RiskWeights(.5, .5, .5, .5))
