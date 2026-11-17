"""MAPPO building blocks; complete CybORG training is not implemented yet."""
import torch


def clipped_policy_loss(new_log_probs, old_log_probs, advantages, clip_epsilon=0.2):
    if not 0 <= clip_epsilon < 1:
        raise ValueError("clip_epsilon must be in [0, 1)")
    ratios = torch.exp(new_log_probs - old_log_probs)
    return -torch.minimum(
        ratios * advantages,
        torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages,
    ).mean()


def value_loss(values, returns):
    return 0.5 * (values - returns).square().mean()


def normalize_advantages(advantages, eps=1e-8):
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95,
                bootstrap_value=None):
    """Compute GAE for one trajectory.

    rewards, values and dones are aligned 1-D tensors of length T.
    dones[t] marks an actual terminal transition (not a time-limit truncation).
    For nonterminal rollouts, bootstrap_value is V(s_T). Never bootstrap
    terminal transitions. Returns (advantages, lambda_returns).
    """
    if rewards.ndim != 1 or values.ndim != 1 or dones.ndim != 1:
        raise ValueError("Expected 1-D rewards, values and dones")
    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("rewards, values and dones lengths must match")
    if not (0 <= gamma <= 1 and 0 <= gae_lambda <= 1):
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    next_value = (torch.as_tensor(0.0, dtype=values.dtype, device=values.device)
                  if bootstrap_value is None else
                  torch.as_tensor(bootstrap_value, dtype=values.dtype, device=values.device))
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t].to(rewards.dtype)
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages, advantages + values
