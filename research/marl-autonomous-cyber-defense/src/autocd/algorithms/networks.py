import torch
from torch import nn
from torch.distributions import Categorical

def mlp(input_dim, output_dim, hidden_dim=128):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim))

class MultiAgentActorCritic(nn.Module):
    """Decentralized per-agent actors plus centralized training critic."""
    def __init__(self, agent_names, observation_dim, action_dim, global_state_dim,
                 hidden_dim=128):
        super().__init__()
        self.agent_names = list(agent_names)
        self.actors = nn.ModuleDict({n: mlp(observation_dim, action_dim, hidden_dim)
                                     for n in self.agent_names})
        self.critic = mlp(global_state_dim, 1, hidden_dim)

    def act(self, agent, observation, action_mask=None):
        logits = self.actors[agent](observation)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(self, agent, observations, actions, action_masks=None):
        logits = self.actors[agent](observations)
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks.bool(), torch.finfo(logits.dtype).min)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()

    def value(self, global_state):
        return self.critic(global_state).squeeze(-1)
