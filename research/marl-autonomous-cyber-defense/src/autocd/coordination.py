from dataclasses import dataclass

@dataclass(frozen=True)
class TaskAssignment:
    agent: str
    task: str
    target: str | None
    priority: float

class RiskAwareCoordinator:
    """Interpretable Phase-1 priority baseline, not learned hierarchical RL."""
    def assign(self, risk_by_asset, available_agents=("detection", "risk", "response")):
        if any(not 0 <= float(v) <= 1 for v in risk_by_asset.values()):
            raise ValueError("Risk values must be in [0, 1].")
        if not available_agents:
            return []
        ranked = sorted(risk_by_asset.items(), key=lambda x: x[1], reverse=True)
        top = ranked[0] if ranked else (None, 0.0)
        out = []
        if "detection" in available_agents:
            out.append(TaskAssignment("detection", "investigate", top[0], top[1]))
        if "risk" in available_agents:
            out.append(TaskAssignment("risk", "assess", top[0], top[1]))
        if "response" in available_agents and top[0] is not None and top[1] >= .67:
            out.append(TaskAssignment("response", "recommend_containment", top[0], top[1]))
        return out
