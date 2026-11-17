"""Cooperative MAPPO trainer for parallel multi-agent environments.

Validated against MockEnterpriseEnv in unit tests. CybORG execution must be
validated separately with the pinned CC4 adapter and documented action masks.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class MAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 128


class CTDEActorCritic(nn.Module):
    """Per-agent decentralized actors and a centralized scalar team critic."""
    def __init__(self, observation_dims, action_dims, global_state_dim, hidden_dim=128):
        super().__init__()
        if set(observation_dims) != set(action_dims):
            raise ValueError("observation_dims and action_dims must have identical agents")
        self.agents = tuple(observation_dims)
        self.actors = nn.ModuleDict({
            a: nn.Sequential(nn.Linear(observation_dims[a], hidden_dim), nn.Tanh(),
                             nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                             nn.Linear(hidden_dim, action_dims[a]))
            for a in self.agents
        })
        self.critic = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def distribution(self, agent, obs, mask=None):
        logits = self.actors[agent](obs)
        if mask is not None:
            mask = mask.bool()
            if not mask.any(dim=-1).all():
                raise ValueError("Each action mask row must contain a valid action")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return Categorical(logits=logits)

    def value(self, global_state):
        return self.critic(global_state).squeeze(-1)


class MAPPOTrainer:
    """Train MAPPO on a parallel-style env exposing reset/step and agent list."""
    def __init__(self, env, config=None, *, seed=42, device="cpu",
                 reward_aggregation="sum"):
        if reward_aggregation not in {"sum", "mean"}:
            raise ValueError("reward_aggregation must be 'sum' or 'mean'")
        self.env = env
        self.cfg = config or MAPPOConfig()
        self.device = torch.device(device)
        self.seed, self.episode_index = int(seed), 0
        self.reward_aggregation = reward_aggregation
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        obs, infos = env.reset(seed=seed)
        self.agents = tuple(env.possible_agents)
        if not self.agents:
            raise ValueError("Environment must define possible_agents")
        self.observation_dims = {a: int(np.asarray(obs[a]).size) for a in self.agents}
        if len(set(self.observation_dims.values())) != 1:
            raise ValueError("This trainer requires equal observation dimensions (pad spaces)")
        self.action_dims = {}
        for a in self.agents:
            if hasattr(env, "action_dim"):
                self.action_dims[a] = int(env.action_dim)
            elif hasattr(env, "action_dims"):
                self.action_dims[a] = int(env.action_dims[a])
            else:
                raise ValueError("Environment must expose action_dim or action_dims")
        self.global_state_dim = sum(self.observation_dims.values())
        self.model = CTDEActorCritic(self.observation_dims, self.action_dims,
                                     self.global_state_dim, self.cfg.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        self._obs = obs
        self._infos = infos

    def _state(self, observations):
        return np.concatenate([np.asarray(observations[a], dtype=np.float32).reshape(-1)
                               for a in self.agents])

    def _mask(self, agent, info):
        if "action_mask" in info.get(agent, {}):
            m = np.asarray(info[agent]["action_mask"], dtype=bool).reshape(-1)
        elif hasattr(self.env, "action_masks"):
            m = np.asarray(self.env.action_masks()[agent], dtype=bool).reshape(-1)
        else:
            m = np.ones(self.action_dims[agent], dtype=bool)
        if m.size != self.action_dims[agent] or not m.any():
            raise ValueError(f"Invalid action mask for {agent}")
        return m

    def collect_rollout(self, steps=128):
        if steps < 1:
            raise ValueError("steps must be positive")
        rows = {k: [] for k in ("states", "rewards", "terminated", "boundaries", "next_values", "values",
                                "obs", "actions", "logps", "masks")}
        episode_returns, current_return = [], 0.0
        for _ in range(steps):
            state = self._state(self._obs)
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action_dict, obs_row, action_row, logp_row, mask_row = {}, {}, {}, {}, {}
            with torch.no_grad():
                value = self.model.value(state_t)
                for a in self.agents:
                    ob = torch.as_tensor(self._obs[a], dtype=torch.float32, device=self.device)
                    mask = self._mask(a, self._infos)
                    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
                    dist = self.model.distribution(a, ob, mask_t)
                    act = torch.multinomial(dist.probs, 1, generator=self.generator).squeeze(-1)
                    action_dict[a] = int(act.item())
                    obs_row[a] = np.asarray(self._obs[a], dtype=np.float32)
                    action_row[a] = int(act.item())
                    logp_row[a] = float(dist.log_prob(act).item())
                    mask_row[a] = mask
            next_obs, rewards, terms, truncs, next_infos = self.env.step(action_dict)
            term_flags = [bool(terms.get(a, False)) for a in self.agents]
            trunc_flags = [bool(truncs.get(a, False)) for a in self.agents]
            done_flags = [t or tr for t, tr in zip(term_flags, trunc_flags)]
            if any(done_flags) and not all(done_flags):
                raise RuntimeError("Partial agent termination is unsupported for shared team critic")
            terminated_team = all(term_flags)
            boundary = all(done_flags)
            team_reward = float(sum(rewards.get(a, 0.0) for a in self.agents))
            if self.reward_aggregation == "mean":
                team_reward /= len(self.agents)
            current_return += team_reward
            if terminated_team:
                next_value = 0.0
            else:
                if set(next_obs) != set(self.agents):
                    if boundary:
                        raise RuntimeError("Truncated episode must return final observations for bootstrapping")
                    raise RuntimeError("Environment omitted an agent observation mid-episode")
                next_state = self._state(next_obs)
                with torch.no_grad():
                    next_value = float(self.model.value(torch.as_tensor(
                        next_state, dtype=torch.float32, device=self.device)).item())
            rows["states"].append(state)
            rows["rewards"].append(team_reward)
            rows["terminated"].append(float(terminated_team))
            rows["boundaries"].append(float(boundary))
            rows["next_values"].append(next_value)
            rows["values"].append(float(value.item()))
            rows["obs"].append(obs_row)
            rows["actions"].append(action_row)
            rows["logps"].append(logp_row)
            rows["masks"].append(mask_row)
            self._obs, self._infos = next_obs, next_infos
            if boundary:
                episode_returns.append(current_return)
                current_return = 0.0
                self.episode_index += 1
                self._obs, self._infos = self.env.reset(seed=self.seed + self.episode_index)
        rows["bootstrap"] = rows["next_values"][-1]
        rows["episode_returns"] = episode_returns
        return rows

    def update(self, rollout):
        rewards = torch.tensor(rollout["rewards"], dtype=torch.float32, device=self.device)
        values = torch.tensor(rollout["values"], dtype=torch.float32, device=self.device)
        terminated = torch.tensor(rollout["terminated"], dtype=torch.float32, device=self.device)
        boundaries = torch.tensor(rollout["boundaries"], dtype=torch.float32, device=self.device)
        next_values = torch.tensor(rollout["next_values"], dtype=torch.float32, device=self.device)
        advantages = torch.zeros_like(rewards)
        carry = torch.zeros((), dtype=torch.float32, device=self.device)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.cfg.gamma * next_values[t] * (1.0 - terminated[t]) - values[t]
            carry = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - boundaries[t]) * carry
            advantages[t] = carry
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        states = torch.as_tensor(np.asarray(rollout["states"]), dtype=torch.float32, device=self.device)
        n = len(rewards)
        obs_t = {a: torch.as_tensor(np.stack([r[a] for r in rollout["obs"]]),
                                    dtype=torch.float32, device=self.device) for a in self.agents}
        actions_t = {a: torch.tensor([r[a] for r in rollout["actions"]],
                                     dtype=torch.long, device=self.device) for a in self.agents}
        old_logps_t = {a: torch.tensor([r[a] for r in rollout["logps"]],
                                       dtype=torch.float32, device=self.device) for a in self.agents}
        masks_t = {a: torch.as_tensor(np.stack([r[a] for r in rollout["masks"]]),
                                      dtype=torch.bool, device=self.device) for a in self.agents}
        actor_losses, critic_losses, entropies = [], [], []
        for _ in range(self.cfg.update_epochs):
            permutation = torch.randperm(n, device=self.device, generator=self.generator)
            for start in range(0, n, self.cfg.minibatch_size):
                idx = permutation[start:start + self.cfg.minibatch_size]
                critic_values = self.model.value(states[idx])
                critic_loss = 0.5 * (critic_values - returns[idx]).pow(2).mean()
                actor_loss_sum = torch.zeros((), device=self.device)
                entropy_sum = torch.zeros((), device=self.device)
                for a in self.agents:
                    dist = self.model.distribution(a, obs_t[a][idx], masks_t[a][idx])
                    new_logp = dist.log_prob(actions_t[a][idx])
                    ratio = torch.exp(new_logp - old_logps_t[a][idx])
                    s1 = ratio * advantages[idx]
                    s2 = torch.clamp(ratio, 1-self.cfg.clip_epsilon,
                                     1+self.cfg.clip_epsilon) * advantages[idx]
                    actor_loss_sum = actor_loss_sum - torch.minimum(s1, s2).mean() / len(self.agents)
                    entropy_sum = entropy_sum + dist.entropy().mean() / len(self.agents)
                loss = actor_loss_sum + self.cfg.value_coef * critic_loss - self.cfg.entropy_coef * entropy_sum
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                actor_losses.append(float(actor_loss_sum.detach()))
                critic_losses.append(float(critic_loss.detach()))
                entropies.append(float(entropy_sum.detach()))
        return {"actor_loss": float(np.mean(actor_losses)),
                "critic_loss": float(np.mean(critic_losses)),
                "entropy": float(np.mean(entropies))}

    def train(self, total_steps=1024, rollout_steps=128):
        if total_steps < 1 or rollout_steps < 1:
            raise ValueError("step counts must be positive")
        history = []
        steps = 0
        while steps < total_steps:
            rollout = self.collect_rollout(min(rollout_steps, total_steps - steps))
            metrics = self.update(rollout)
            steps += len(rollout["rewards"])
            history.append({"environment_steps": steps, **metrics,
                            "episode_returns": rollout["episode_returns"]})
        return history
