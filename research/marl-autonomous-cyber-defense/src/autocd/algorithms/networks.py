"""Decentralized discrete actors and a centralized scalar critic."""
import torch
from torch import nn
from torch.distributions import Categorical


def mlp(input_dim, output_dim, hidden_dim=128):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class MultiAgentActorCritic(nn.Module):
    def __init__(self, agent_names, observation_dim, action_dim, global_state_dim,
                 hidden_dim=128):
        super().__init__()
        self.agent_names = list(agent_names)
        if len(set(self.agent_names)) != len(self.agent_names):
            raise ValueError("agent_names must be unique")
        self.actors = nn.ModuleDict({
            name: mlp(observation_dim, action_dim, hidden_dim)
            for name in self.agent_names
        })
        self.critic = mlp(global_state_dim, 1, hidden_dim)

    def _distribution(self, agent, observations, action_mask=None):
        logits = self.actors[agent](observations)
        if action_mask is not None:
            mask = action_mask.to(device=logits.device, dtype=torch.bool)
            if mask.shape != logits.shape:
                raise ValueError("Action mask must match logits shape")
            if not torch.all(mask.any(dim=-1)):
                raise ValueError("Every action row needs at least one valid action")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return Categorical(logits=logits)

    def act(self, agent, observation, action_mask=None):
        dist = self._distribution(agent, observation, action_mask)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(self, agent, observations, actions, action_masks=None):
        dist = self._distribution(agent, observations, action_masks)
        return dist.log_prob(actions), dist.entropy()

    def value(self, global_state):
        return self.critic(global_state).squeeze(-1)
