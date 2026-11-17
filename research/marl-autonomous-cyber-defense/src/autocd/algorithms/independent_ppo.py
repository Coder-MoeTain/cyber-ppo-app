"""Independent PPO baseline with a separate actor and critic per agent."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class IPPOConfig:
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


class IndependentActorCritic(nn.Module):
    """Each agent has a local actor and a local value function (no CTDE)."""
    def __init__(self, observation_dims, action_dims, hidden_dim=128):
        super().__init__()
        if set(observation_dims) != set(action_dims):
            raise ValueError("observation/action agent keys must match")
        self.agents = tuple(observation_dims)
        self.actors = nn.ModuleDict({
            a: nn.Sequential(nn.Linear(observation_dims[a], hidden_dim), nn.Tanh(),
                             nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                             nn.Linear(hidden_dim, action_dims[a]))
            for a in self.agents
        })
        self.critics = nn.ModuleDict({
            a: nn.Sequential(nn.Linear(observation_dims[a], hidden_dim), nn.Tanh(),
                             nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                             nn.Linear(hidden_dim, 1))
            for a in self.agents
        })

    def distribution(self, agent, observations, mask=None):
        logits = self.actors[agent](observations)
        if mask is not None:
            mask = mask.to(device=logits.device, dtype=torch.bool)
            if logits.ndim == 2 and mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != logits.shape or not mask.any(dim=-1).all():
                raise ValueError(f"Invalid action mask for {agent}")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return Categorical(logits=logits)

    def value(self, agent, observations):
        return self.critics[agent](observations).squeeze(-1)


class IndependentPPOTrainer:
    """Independent learners; each agent optimizes only its own reward stream."""
    def __init__(self, env, config=None, *, seed=42, device="cpu"):
        self.env, self.cfg = env, config or IPPOConfig()
        self.seed, self.episode_index = int(seed), 0
        self.device = torch.device(device)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        obs, infos = env.reset(seed=self.seed)
        self.agents = tuple(env.possible_agents)
        if not self.agents or set(obs) != set(self.agents):
            raise ValueError("reset must return all possible agent observations")
        self.observation_dims = {a: int(np.asarray(obs[a]).size) for a in self.agents}
        self.action_dims = {}
        for a in self.agents:
            if hasattr(env, "action_dims") and a in env.action_dims:
                self.action_dims[a] = int(env.action_dims[a])
            elif hasattr(env, "action_dim"):
                self.action_dims[a] = int(env.action_dim)
            else:
                raise ValueError(f"Missing action dimension for {a}")
        self.model = IndependentActorCritic(
            self.observation_dims, self.action_dims, self.cfg.hidden_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        self._obs, self._infos = obs, infos or {}

    def _mask(self, agent):
        if "action_mask" in self._infos.get(agent, {}):
            mask = np.asarray(self._infos[agent]["action_mask"], dtype=bool).reshape(-1)
        elif hasattr(self.env, "action_masks"):
            mask = np.asarray(self.env.action_masks()[agent], dtype=bool).reshape(-1)
        else:
            mask = np.ones(self.action_dims[agent], dtype=bool)
        if mask.size != self.action_dims[agent] or not mask.any():
            raise ValueError(f"Invalid action mask for {agent}")
        return mask

    def collect_rollout(self, steps=128):
        if steps < 1:
            raise ValueError("steps must be positive")
        rows = {a: {k: [] for k in (
            "obs", "actions", "logps", "masks", "rewards", "values",
            "next_values", "terminated", "boundaries",
        )} for a in self.agents}
        episode_returns = {a: 0.0 for a in self.agents}
        completed_returns = {a: [] for a in self.agents}

        for _ in range(steps):
            action_dict = {}
            observations, masks, values, logps = {}, {}, {}, {}
            with torch.no_grad():
                for agent in self.agents:
                    ob = np.asarray(self._obs[agent], dtype=np.float32).reshape(-1)
                    mask = self._mask(agent)
                    obs_t = torch.as_tensor(ob, dtype=torch.float32, device=self.device)
                    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
                    dist = self.model.distribution(agent, obs_t, mask_t)
                    action_t = torch.multinomial(dist.probs, 1, generator=self.generator).squeeze(-1)
                    action_dict[agent] = int(action_t.item())
                    observations[agent], masks[agent] = ob, mask
                    values[agent] = float(self.model.value(agent, obs_t).item())
                    logps[agent] = float(dist.log_prob(action_t).item())

            next_obs, rewards, terms, truncs, next_infos = self.env.step(action_dict)
            done_flags = [bool(terms.get(a, False) or truncs.get(a, False)) for a in self.agents]
            if any(done_flags) and not all(done_flags):
                raise RuntimeError("Partial agent termination is unsupported in this IPPO adapter")
            for agent in self.agents:
                terminated = bool(terms.get(agent, False))
                boundary = bool(terms.get(agent, False) or truncs.get(agent, False))
                reward = float(rewards.get(agent, 0.0))
                if terminated:
                    next_value = 0.0
                else:
                    if agent not in next_obs:
                        raise RuntimeError(f"Missing final observation for nonterminal {agent}")
                    nxt = torch.as_tensor(next_obs[agent], dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        next_value = float(self.model.value(agent, nxt).item())
                row = rows[agent]
                row["obs"].append(observations[agent])
                row["actions"].append(action_dict[agent])
                row["logps"].append(logps[agent])
                row["masks"].append(masks[agent])
                row["rewards"].append(reward)
                row["values"].append(values[agent])
                row["next_values"].append(next_value)
                row["terminated"].append(float(terminated))
                row["boundaries"].append(float(boundary))
                episode_returns[agent] += reward
            self._obs, self._infos = next_obs, next_infos or {}
            if all(done_flags):
                for agent in self.agents:
                    completed_returns[agent].append(episode_returns[agent])
                    episode_returns[agent] = 0.0
                self.episode_index += 1
                self._obs, self._infos = self.env.reset(seed=self.seed + self.episode_index)

        return {"agents": rows, "episode_returns": completed_returns}

    def update(self, rollout):
        losses = []
        metrics = {}
        for agent in self.agents:
            row = rollout["agents"][agent]
            rewards = torch.tensor(row["rewards"], dtype=torch.float32, device=self.device)
            values = torch.tensor(row["values"], dtype=torch.float32, device=self.device)
            next_values = torch.tensor(row["next_values"], dtype=torch.float32, device=self.device)
            terminated = torch.tensor(row["terminated"], dtype=torch.float32, device=self.device)
            boundaries = torch.tensor(row["boundaries"], dtype=torch.float32, device=self.device)
            adv = torch.zeros_like(rewards)
            carry = torch.zeros((), dtype=torch.float32, device=self.device)
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.cfg.gamma * next_values[t] * (1-terminated[t]) - values[t]
                carry = delta + self.cfg.gamma * self.cfg.gae_lambda * (1-boundaries[t]) * carry
                adv[t] = carry
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            obs = torch.as_tensor(np.stack(row["obs"]), dtype=torch.float32, device=self.device)
            actions = torch.tensor(row["actions"], dtype=torch.long, device=self.device)
            old_logps = torch.tensor(row["logps"], dtype=torch.float32, device=self.device)
            masks = torch.as_tensor(np.stack(row["masks"]), dtype=torch.bool, device=self.device)
            actor_losses, critic_losses, entropies = [], [], []
            n = len(rewards)
            for _ in range(self.cfg.update_epochs):
                order = torch.randperm(n, device=self.device, generator=self.generator)
                for start in range(0, n, self.cfg.minibatch_size):
                    idx = order[start:start+self.cfg.minibatch_size]
                    dist = self.model.distribution(agent, obs[idx], masks[idx])
                    new_logps = dist.log_prob(actions[idx])
                    ratio = torch.exp(new_logps - old_logps[idx])
                    clipped = torch.clamp(ratio, 1-self.cfg.clip_epsilon,
                                          1+self.cfg.clip_epsilon) * adv[idx]
                    actor_loss = -torch.minimum(ratio*adv[idx], clipped).mean()
                    value_pred = self.model.value(agent, obs[idx])
                    critic_loss = 0.5 * (value_pred - returns[idx]).square().mean()
                    entropy = dist.entropy().mean()
                    loss = actor_loss + self.cfg.value_coef*critic_loss - self.cfg.entropy_coef*entropy
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()
                    actor_losses.append(float(actor_loss.detach()))
                    critic_losses.append(float(critic_loss.detach()))
                    entropies.append(float(entropy.detach()))
            metrics[agent] = {
                "actor_loss": float(np.mean(actor_losses)),
                "critic_loss": float(np.mean(critic_losses)),
                "entropy": float(np.mean(entropies)),
            }
            losses.extend(actor_losses)
        return {"agents": metrics, "mean_actor_loss": float(np.mean(losses))}

    def train(self, total_steps=1024, rollout_steps=128):
        if total_steps < 1 or rollout_steps < 1:
            raise ValueError("step counts must be positive")
        history, steps = [], 0
        while steps < total_steps:
            rollout = self.collect_rollout(min(rollout_steps, total_steps-steps))
            metrics = self.update(rollout)
            steps += len(next(iter(rollout["agents"].values()))["rewards"])
            history.append({"environment_steps": steps, **metrics})
        return history
