"""Minimal on-policy multi-agent PPO trainer for interface validation.

This trainer intentionally uses the MOCK environment. Its results are not
CybORG benchmarks. For research comparisons, add a version-pinned CybORG
adapter, matched-budget baselines and a learned hierarchical coordinator.
"""
from dataclasses import dataclass
import random

import numpy as np
import torch

from autocd.algorithms.mappo import (
    clipped_policy_loss, compute_gae, normalize_advantages, value_loss,
)
from autocd.algorithms.networks import MultiAgentActorCritic
from autocd.envs.mock_enterprise import MockEnterpriseEnv


@dataclass
class TrainConfig:
    episodes: int = 20
    seed: int = 42
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    update_epochs: int = 4
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5


def train_mock(config=None):
    """Return trained model and episode-return history (mock only)."""
    cfg = config or TrainConfig()
    if cfg.episodes < 1 or cfg.update_epochs < 1:
        raise ValueError("episodes and update_epochs must be positive")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = MockEnterpriseEnv(seed=cfg.seed)
    names = env.possible_agents
    state_dim = env.observation_dim * len(names)
    model = MultiAgentActorCritic(names, env.observation_dim, env.action_dim, state_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    history = []

    for episode in range(cfg.episodes):
        obs, _ = env.reset(seed=cfg.seed + episode)
        trajectory = []
        episode_return = 0.0
        while env.agents:
            state = torch.tensor(
                np.concatenate([obs[name] for name in names]), dtype=torch.float32
            )
            with torch.no_grad():
                value = model.value(state).detach()
                actions = {}
                log_probs = {}
                for name in names:
                    observation = torch.tensor(obs[name], dtype=torch.float32)
                    action, log_prob, _ = model.act(name, observation)
                    actions[name] = int(action.item())
                    log_probs[name] = log_prob.detach()
            nxt, rewards, terms, truncs, _ = env.step(actions)
            reward = float(np.mean(list(rewards.values())))
            episode_return += reward
            trajectory.append((obs, state, actions, log_probs, value, reward,
                               bool(all(terms.values()))))
            obs = nxt

        rewards_t = torch.tensor([x[5] for x in trajectory], dtype=torch.float32)
        values_t = torch.stack([x[4] for x in trajectory])
        dones_t = torch.tensor([x[6] for x in trajectory], dtype=torch.bool)
        advantages, returns = compute_gae(
            rewards_t, values_t, dones_t, cfg.gamma, cfg.gae_lambda
        )
        advantages = normalize_advantages(advantages).detach()
        returns = returns.detach()
        states = torch.stack([x[1] for x in trajectory])
        old_log_probs = {
            name: torch.stack([x[3][name] for x in trajectory]) for name in names
        }
        observations = {
            name: torch.tensor(
                np.stack([x[0][name] for x in trajectory]), dtype=torch.float32
            ) for name in names
        }
        actions_t = {
            name: torch.tensor([x[2][name] for x in trajectory], dtype=torch.long)
            for name in names
        }

        for _ in range(cfg.update_epochs):
            policy_losses = []
            entropies = []
            for name in names:
                new_log_probs, entropy = model.evaluate_actions(
                    name, observations[name], actions_t[name]
                )
                policy_losses.append(clipped_policy_loss(
                    new_log_probs, old_log_probs[name], advantages, cfg.clip_epsilon
                ))
                entropies.append(entropy.mean())
            critic_loss = value_loss(model.value(states), returns)
            loss = (
                torch.stack(policy_losses).mean()
                + cfg.value_coefficient * critic_loss
                - cfg.entropy_coefficient * torch.stack(entropies).mean()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        history.append(episode_return)

    env.close()
    return model, history
