"""Supervised local-observation risk estimator for CC4.

Ground-truth red-session labels may be used only to create offline training
labels. At inference, the estimator receives only the acting Blue agent's
permitted observation vector and agent identity.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


class RiskEstimatorNet(nn.Module):
    def __init__(self, observation_dim: int, num_agents: int, hidden_dim: int = 128):
        super().__init__()
        if min(observation_dim, num_agents, hidden_dim) < 1:
            raise ValueError("model dimensions must be positive")
        self.observation_dim = observation_dim
        self.num_agents = num_agents
        self.net = nn.Sequential(
            nn.Linear(observation_dim + num_agents, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, observations: torch.Tensor, agent_ids: torch.Tensor):
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        agent_ids = agent_ids.to(device=observations.device, dtype=torch.long).reshape(-1)
        if observations.shape[1] != self.observation_dim or agent_ids.numel() != observations.shape[0]:
            raise ValueError("observation or agent-id shape mismatch")
        if torch.any((agent_ids < 0) | (agent_ids >= self.num_agents)):
            raise ValueError("agent id outside configured range")
        one_hot = torch.nn.functional.one_hot(agent_ids, self.num_agents).to(observations.dtype)
        return self.net(torch.cat((observations.float(), one_hot), dim=-1)).squeeze(-1)


def binary_metrics(labels, probabilities, bins=10):
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.size != p.size or y.size == 0 or not np.isfinite(p).all():
        raise ValueError("labels/probabilities must be non-empty, finite and equal length")
    if np.any((y != 0) & (y != 1)) or np.any((p < 0) | (p > 1)):
        raise ValueError("labels must be binary and probabilities in [0,1]")
    brier = float(np.mean((p - y) ** 2))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if mask.any():
            ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    positives, negatives = int(y.sum()), int((1 - y).sum())
    if positives and negatives:
        order = np.argsort(p, kind="mergesort")
        sorted_p = p[order]
        ranks = np.empty(y.size, dtype=np.float64)
        start = 0
        while start < y.size:
            end = start + 1
            while end < y.size and sorted_p[end] == sorted_p[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        auc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2)
                    / (positives * negatives))
        descending = np.argsort(-p, kind="mergesort")
        yd = y[descending]
        precision = np.cumsum(yd) / np.arange(1, y.size + 1)
        ap = float(precision[yd == 1].mean())
    else:
        auc, ap = float("nan"), float("nan")
    return {"brier": brier, "ece": float(ece), "auroc": auc, "auprc": ap,
            "positive_rate": float(y.mean()), "n": int(y.size)}


class RiskPredictor:
    """Callable risk_fn adapter compatible with RiskAwareHierarchicalTrainer."""
    def __init__(self, model, agent_names, temperature=1.0, device="cpu"):
        self.model = model.to(device).eval()
        self.agent_names = tuple(agent_names)
        self.agent_to_id = {name: i for i, name in enumerate(self.agent_names)}
        self.temperature = float(temperature)
        self.device = torch.device(device)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = RiskEstimatorNet(
            checkpoint["observation_dim"], len(checkpoint["agent_names"]),
            checkpoint["hidden_dim"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, checkpoint["agent_names"], checkpoint["temperature"], device)

    def __call__(self, agent, observation, info=None):
        if agent not in self.agent_to_id:
            raise KeyError(f"Unknown agent for risk model: {agent}")
        x = torch.as_tensor(np.asarray(observation, dtype=np.float32),
                            dtype=torch.float32, device=self.device)
        aid = torch.tensor([self.agent_to_id[agent]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logit = self.model(x, aid) / self.temperature
            return float(torch.sigmoid(logit).item())
