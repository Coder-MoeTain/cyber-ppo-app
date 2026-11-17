"""Risk-aware hierarchical MAPPO trainer.

This trainer optimizes per-agent manager and worker policies with clipped PPO
objectives and a centralized team critic. A domain-validated risk_fn is required;
the trainer never invents a risk score from arbitrary simulator observations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from autocd.algorithms.hierarchical import RiskAwareHierarchicalActorCritic


@dataclass
class RAHMAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 64
    manager_entropy_coef: float = 0.01
    worker_entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    num_tasks: int = 6


class RiskAwareHierarchicalTrainer:
    """Train RA-HMAPPO on a parallel multi-agent environment.

    risk_fn(agent, observation, info) must return a calibrated/normalized scalar
    in [0, 1]. The default team reward is the sum of agent rewards, matching a
    cooperative team-utility objective; report per-agent metrics separately.
    """

    def __init__(
        self,
        env,
        risk_fn: Callable,
        config: RAHMAPPOConfig | None = None,
        *,
        seed: int = 42,
        device: str = "cpu",
        reward_aggregation: str = "sum",
        task_action_mapper: Callable | None = None,
    ):
        if not callable(risk_fn):
            raise TypeError("risk_fn is required and must be callable")
        if reward_aggregation not in {"sum", "mean"}:
            raise ValueError("reward_aggregation must be 'sum' or 'mean'")
        self.env, self.risk_fn = env, risk_fn
        self.cfg = config or RAHMAPPOConfig()
        self.seed, self.episode_index = int(seed), 0
        self.device = torch.device(device)
        self.reward_aggregation = reward_aggregation
        self.task_action_mapper = task_action_mapper
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        observations, infos = self.env.reset(seed=self.seed)
        self.agents = tuple(self.env.possible_agents)
        if not self.agents or set(observations) != set(self.agents):
            raise ValueError("reset must return observations for every possible agent")
        self.observation_dims = {a: int(np.asarray(observations[a]).size) for a in self.agents}
        if len(set(self.observation_dims.values())) != 1:
            raise ValueError("Hierarchical trainer requires equal padded observation dimensions")
        self.action_dims = {}
        for agent in self.agents:
            if hasattr(env, "action_dims") and agent in env.action_dims:
                self.action_dims[agent] = int(env.action_dims[agent])
            elif hasattr(env, "action_dim"):
                self.action_dims[agent] = int(env.action_dim)
            else:
                raise ValueError(f"Environment does not expose action dimension for {agent}")
        self.obs_dim = next(iter(self.observation_dims.values()))
        self.global_state_dim = sum(self.observation_dims.values())
        self.model = RiskAwareHierarchicalActorCritic(
            self.agents, self.obs_dim, max(self.action_dims.values()),
            self.global_state_dim, self.cfg.num_tasks, self.cfg.hidden_dim
        ).to(self.device)
        if len(set(self.action_dims.values())) != 1:
            raise ValueError("This initial trainer requires padded equal action dimensions")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        self._obs, self._infos = observations, infos or {}

    def _state(self, observations):
        return np.concatenate([
            np.asarray(observations[a], dtype=np.float32).reshape(-1) for a in self.agents
        ])

    def _risk(self, agent, observation, info):
        value = float(self.risk_fn(agent, observation, info))
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"risk_fn returned invalid score for {agent}: {value}")
        return value

    def _action_mask(self, agent):
        info = self._infos.get(agent, {})
        if "action_mask" in info:
            mask = np.asarray(info["action_mask"], dtype=bool).reshape(-1)
        elif hasattr(self.env, "action_masks"):
            mask = np.asarray(self.env.action_masks()[agent], dtype=bool).reshape(-1)
        else:
            mask = np.ones(self.action_dims[agent], dtype=bool)
        if mask.size != self.action_dims[agent] or not mask.any():
            raise ValueError(f"Invalid action mask for {agent}")
        return mask

    def collect_rollout(self, steps: int = 128):
        if steps < 1:
            raise ValueError("steps must be positive")
        keys = ("states", "rewards", "terminated", "boundaries", "next_values",
                "values", "obs", "risks", "tasks", "actions", "manager_logps",
                "worker_logps", "action_masks", "task_masks", "episode_returns", "agent_rewards")
        rows = {key: [] for key in keys}
        running_return = 0.0

        for _ in range(steps):
            state = self._state(self._obs)
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            observations, risks, tasks, actions = {}, {}, {}, {}
            manager_logps, worker_logps, masks, task_masks = {}, {}, {}, {}
            with torch.no_grad():
                value = float(self.model.value(state_tensor).item())
                action_dict = {}
                for agent in self.agents:
                    obs = np.asarray(self._obs[agent], dtype=np.float32).reshape(-1)
                    info = self._infos.get(agent, {})
                    risk = self._risk(agent, obs, info)
                    mask = self._action_mask(agent)
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    risk_tensor = torch.tensor([risk], dtype=torch.float32, device=self.device)
                    if self.task_action_mapper is None:
                        task_action_masks = np.repeat(mask[None, :], self.cfg.num_tasks, axis=0)
                    else:
                        task_action_masks = np.asarray(
                            self.task_action_mapper(agent, obs, info, mask), dtype=bool
                        )
                    if task_action_masks.shape != (self.cfg.num_tasks, self.action_dims[agent]):
                        raise ValueError(f"Task/action mask shape mismatch for {agent}")
                    task_mask = task_action_masks.any(axis=1)
                    if not task_mask.any():
                        raise ValueError(f"No available high-level task for {agent}")
                    manager_dist = self.model.manager_distribution(
                        agent, obs_tensor, risk_tensor,
                        torch.as_tensor(task_mask, dtype=torch.bool, device=self.device),
                    )
                    task_tensor = torch.multinomial(
                        manager_dist.probs, 1, generator=self.generator
                    ).squeeze(-1)
                    selected_task = int(task_tensor.item())
                    worker_mask = task_action_masks[selected_task]
                    worker_dist = self.model.worker_distribution(
                        agent, obs_tensor, risk_tensor, task_tensor,
                        torch.as_tensor(worker_mask, dtype=torch.bool, device=self.device),
                    )
                    action_tensor = torch.multinomial(
                        worker_dist.probs, 1, generator=self.generator
                    ).squeeze(-1)
                    task, action = int(task_tensor.item()), int(action_tensor.item())
                    observations[agent], risks[agent] = obs, risk
                    tasks[agent], actions[agent] = task, action
                    manager_logps[agent] = float(manager_dist.log_prob(task_tensor).item())
                    worker_logps[agent] = float(worker_dist.log_prob(action_tensor).item())
                    masks[agent] = worker_mask
                    task_masks[agent] = task_mask
                    action_dict[agent] = action

            next_obs, rewards, terms, truncs, next_infos = self.env.step(action_dict)
            term_flags = [bool(terms.get(a, False)) for a in self.agents]
            trunc_flags = [bool(truncs.get(a, False)) for a in self.agents]
            done_flags = [t or tr for t, tr in zip(term_flags, trunc_flags)]
            if any(done_flags) and not all(done_flags):
                raise RuntimeError("Partial agent termination is unsupported for shared team critic")
            terminated = all(term_flags)
            boundary = all(done_flags)
            agent_rewards = {a: float(rewards.get(a, 0.0)) for a in self.agents}
            team_reward = sum(agent_rewards.values())
            if self.reward_aggregation == "mean":
                team_reward /= len(self.agents)

            if terminated:
                next_value = 0.0
            else:
                if set(next_obs) != set(self.agents):
                    if boundary:
                        raise RuntimeError("Truncated episode must return final observations for bootstrapping")
                    raise RuntimeError("Environment omitted an agent observation mid-episode")
                next_state = self._state(next_obs)
                with torch.no_grad():
                    next_value = float(self.model.value(torch.as_tensor(
                        next_state, dtype=torch.float32, device=self.device
                    )).item())

            rows["states"].append(state)
            rows["rewards"].append(team_reward)
            rows["terminated"].append(float(terminated))
            rows["boundaries"].append(float(boundary))
            rows["next_values"].append(next_value)
            rows["values"].append(value)
            rows["obs"].append(observations)
            rows["risks"].append(risks)
            rows["tasks"].append(tasks)
            rows["actions"].append(actions)
            rows["manager_logps"].append(manager_logps)
            rows["worker_logps"].append(worker_logps)
            rows["action_masks"].append(masks)
            rows["task_masks"].append(task_masks)
            rows["agent_rewards"].append(agent_rewards)
            running_return += team_reward

            self._obs, self._infos = next_obs, next_infos or {}
            if boundary:
                rows["episode_returns"].append(running_return)
                running_return = 0.0
                self.episode_index += 1
                self._obs, self._infos = self.env.reset(seed=self.seed + self.episode_index)

        return rows

    def update(self, rollout):
        rewards = torch.as_tensor(rollout["rewards"], dtype=torch.float32, device=self.device)
        values = torch.as_tensor(rollout["values"], dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(rollout["terminated"], dtype=torch.float32, device=self.device)
        boundaries = torch.as_tensor(rollout["boundaries"], dtype=torch.float32, device=self.device)
        next_values = torch.as_tensor(rollout["next_values"], dtype=torch.float32, device=self.device)
        if rewards.numel() == 0:
            raise ValueError("Cannot update from an empty rollout")

        advantages = torch.zeros_like(rewards)
        carry = torch.zeros((), dtype=torch.float32, device=self.device)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.cfg.gamma * next_values[t] * (1.0 - terminated[t]) - values[t]
            carry = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - boundaries[t]) * carry
            advantages[t] = carry
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        states = torch.as_tensor(np.asarray(rollout["states"]), dtype=torch.float32, device=self.device)
        obs_t, risks_t, tasks_t, actions_t = {}, {}, {}, {}
        old_manager_t, old_worker_t, masks_t, task_masks_t = {}, {}, {}, {}
        for agent in self.agents:
            obs_t[agent] = torch.as_tensor(
                np.stack([r[agent] for r in rollout["obs"]]), dtype=torch.float32, device=self.device
            )
            risks_t[agent] = torch.tensor(
                [r[agent] for r in rollout["risks"]], dtype=torch.float32, device=self.device
            )
            tasks_t[agent] = torch.tensor(
                [r[agent] for r in rollout["tasks"]], dtype=torch.long, device=self.device
            )
            actions_t[agent] = torch.tensor(
                [r[agent] for r in rollout["actions"]], dtype=torch.long, device=self.device
            )
            old_manager_t[agent] = torch.tensor(
                [r[agent] for r in rollout["manager_logps"]], dtype=torch.float32, device=self.device
            )
            old_worker_t[agent] = torch.tensor(
                [r[agent] for r in rollout["worker_logps"]], dtype=torch.float32, device=self.device
            )
            task_masks_t[agent] = torch.as_tensor(
                np.stack([r[agent] for r in rollout["task_masks"]]),
                dtype=torch.bool, device=self.device
            )
            masks_t[agent] = torch.as_tensor(
                np.stack([r[agent] for r in rollout["action_masks"]]),
                dtype=torch.bool, device=self.device
            )

        actor_losses, critic_losses, manager_entropies, worker_entropies = [], [], [], []
        n = rewards.shape[0]
        for _ in range(self.cfg.update_epochs):
            permutation = torch.randperm(n, device=self.device, generator=self.generator)
            for start in range(0, n, self.cfg.minibatch_size):
                idx = permutation[start:start + self.cfg.minibatch_size]
                value_pred = self.model.value(states[idx])
                critic_loss = 0.5 * (value_pred - returns[idx]).pow(2).mean()
                agent_actor_losses, manager_entropy, worker_entropy = [], [], []
                for agent in self.agents:
                    evaluated = self.model.evaluate_actions(
                        agent, obs_t[agent][idx], risks_t[agent][idx],
                        tasks_t[agent][idx], actions_t[agent][idx],
                        task_masks=task_masks_t[agent][idx],
                        action_masks=masks_t[agent][idx],
                    )
                    new_manager = evaluated["manager_log_prob"]
                    new_worker = evaluated["worker_log_prob"]
                    old_joint = old_manager_t[agent][idx] + old_worker_t[agent][idx]
                    new_joint = new_manager + new_worker
                    ratio = torch.exp(new_joint - old_joint)
                    unclipped = ratio * advantages[idx]
                    clipped = torch.clamp(
                        ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon
                    ) * advantages[idx]
                    agent_actor_losses.append(-torch.minimum(unclipped, clipped).mean())
                    manager_entropy.append(evaluated["manager_entropy"].mean())
                    worker_entropy.append(evaluated["worker_entropy"].mean())

                actor_loss = torch.stack(agent_actor_losses).mean()
                manager_ent = torch.stack(manager_entropy).mean()
                worker_ent = torch.stack(worker_entropy).mean()
                loss = (actor_loss + self.cfg.value_coef * critic_loss
                        - self.cfg.manager_entropy_coef * manager_ent
                        - self.cfg.worker_entropy_coef * worker_ent)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                actor_losses.append(float(actor_loss.detach()))
                critic_losses.append(float(critic_loss.detach()))
                manager_entropies.append(float(manager_ent.detach()))
                worker_entropies.append(float(worker_ent.detach()))

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "manager_entropy": float(np.mean(manager_entropies)),
            "worker_entropy": float(np.mean(worker_entropies)),
        }

    def train(self, total_steps: int = 1024, rollout_steps: int = 128):
        if total_steps < 1 or rollout_steps < 1:
            raise ValueError("step counts must be positive")
        history, steps = [], 0
        while steps < total_steps:
            rollout = self.collect_rollout(min(rollout_steps, total_steps - steps))
            metrics = self.update(rollout)
            steps += len(rollout["rewards"])
            history.append({
                "environment_steps": steps,
                "episode_returns": rollout["episode_returns"],
                **metrics,
            })
        return history
