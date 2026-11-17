"""Risk-conditioned hierarchical policy modules for multi-agent cyber defence."""
from __future__ import annotations
import torch
from torch import nn
from torch.distributions import Categorical


def _mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                         nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                         nn.Linear(hidden_dim, output_dim))


class RiskAwareHierarchicalActorCritic(nn.Module):
    """Decentralized manager/worker policies with a centralized training critic."""
    def __init__(self, agent_names, observation_dim, action_dim, global_state_dim,
                 num_tasks=5, hidden_dim=128):
        super().__init__()
        if not agent_names or len(set(agent_names)) != len(agent_names):
            raise ValueError("agent_names must be non-empty and unique")
        if min(observation_dim, action_dim, global_state_dim, num_tasks, hidden_dim) < 1:
            raise ValueError("All dimensions must be positive")
        self.agent_names = tuple(agent_names)
        self.num_tasks = num_tasks
        self.managers = nn.ModuleDict({
            a: _mlp(observation_dim + 1, num_tasks, hidden_dim) for a in self.agent_names
        })
        self.task_embeddings = nn.Embedding(num_tasks, hidden_dim)
        self.workers = nn.ModuleDict({
            a: _mlp(observation_dim + 1 + hidden_dim, action_dim, hidden_dim)
            for a in self.agent_names
        })
        self.critic = _mlp(global_state_dim, 1, hidden_dim)

    @staticmethod
    def _risk(risk, batch, device):
        risk = risk.reshape(-1, 1).to(device)
        if risk.shape[0] != batch or not torch.isfinite(risk).all() or torch.any((risk < 0) | (risk > 1)):
            raise ValueError("risk must be finite, in [0, 1], and match the batch")
        return risk

    @staticmethod
    def _dist(logits, mask=None):
        if mask is not None:
            mask = mask.to(device=logits.device, dtype=torch.bool)
            if mask.shape != logits.shape or not mask.any(dim=-1).all():
                raise ValueError("Mask must match logits and contain a valid choice in every row")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return Categorical(logits=logits)

    def manager_distribution(self, agent, observation, risk, task_mask=None):
        if agent not in self.managers:
            raise KeyError(f"Unknown agent: {agent}")
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        r = self._risk(risk, observation.shape[0], observation.device)
        return self._dist(self.managers[agent](torch.cat((observation, r), -1)), task_mask)

    def worker_distribution(self, agent, observation, risk, task, action_mask=None):
        if agent not in self.workers:
            raise KeyError(f"Unknown agent: {agent}")
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        batch = observation.shape[0]
        r = self._risk(risk, batch, observation.device)
        task = task.to(device=observation.device, dtype=torch.long).reshape(-1)
        if task.numel() != batch or torch.any((task < 0) | (task >= self.num_tasks)):
            raise ValueError("task indices must be valid and match the batch")
        emb = self.task_embeddings(task)
        logits = self.workers[agent](torch.cat((observation, r, emb), -1))
        return self._dist(logits, action_mask)

    def act(self, agent, observation, risk, task_mask=None, action_mask=None,
            deterministic=False):
        md = self.manager_distribution(agent, observation, risk, task_mask)
        task = md.probs.argmax(-1) if deterministic else md.sample()
        wd = self.worker_distribution(agent, observation, risk, task, action_mask)
        action = wd.probs.argmax(-1) if deterministic else wd.sample()
        return {"task": task, "action": action,
                "manager_log_prob": md.log_prob(task), "worker_log_prob": wd.log_prob(action),
                "manager_entropy": md.entropy(), "worker_entropy": wd.entropy()}

    def evaluate_actions(self, agent, observations, risks, tasks, actions,
                         task_masks=None, action_masks=None):
        md = self.manager_distribution(agent, observations, risks, task_masks)
        wd = self.worker_distribution(agent, observations, risks, tasks, action_masks)
        return {"manager_log_prob": md.log_prob(tasks.long()),
                "worker_log_prob": wd.log_prob(actions.long()),
                "manager_entropy": md.entropy(), "worker_entropy": wd.entropy()}

    def value(self, global_state):
        return self.critic(global_state).squeeze(-1)
