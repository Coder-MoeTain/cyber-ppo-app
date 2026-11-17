from dataclasses import dataclass

@dataclass(frozen=True)
class RiskWeights:
    probability: float = 0.30
    impact: float = 0.30
    exposure: float = 0.20
    criticality: float = 0.20

    def validate(self):
        values = (self.probability, self.impact, self.exposure, self.criticality)
        if any(v < 0 for v in values) or abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("Risk weights must be non-negative and sum to 1.")

@dataclass(frozen=True)
class RiskAssessment:
    probability: float
    impact: float
    exposure: float
    criticality: float
    score: float
    severity: str

class RiskScorer:
    """Transparent weighted risk model; baseline only, not a validated standard."""
    def __init__(self, weights=None):
        self.weights = weights or RiskWeights()
        self.weights.validate()

    def assess(self, probability, impact, exposure, criticality):
        vals = [float(x) for x in (probability, impact, exposure, criticality)]
        if any(x < 0 or x > 1 for x in vals):
            raise ValueError("Risk inputs must be in [0, 1].")
        p, i, e, c = vals
        w = self.weights
        score = w.probability*p + w.impact*i + w.exposure*e + w.criticality*c
        severity = "low" if score < .34 else "medium" if score < .67 else "high"
        return RiskAssessment(p, i, e, c, score, severity)
