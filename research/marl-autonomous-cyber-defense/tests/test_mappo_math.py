import pytest
import torch

from autocd.algorithms.mappo import (
    clipped_policy_loss, compute_gae, normalize_advantages,
)
from autocd.algorithms.networks import MultiAgentActorCritic


def test_terminal_gae_does_not_bootstrap():
    rewards = torch.tensor([1.0])
    values = torch.tensor([0.4])
    dones = torch.tensor([True])
    advantages, returns = compute_gae(
        rewards, values, dones, bootstrap_value=torch.tensor(99.0)
    )
    assert advantages.item() == pytest.approx(0.6)
    assert returns.item() == pytest.approx(1.0)


def test_nonterminal_gae_uses_bootstrap():
    advantages, returns = compute_gae(
        torch.tensor([1.0]), torch.tensor([0.5]),
        torch.tensor([False]), gamma=0.9,
        bootstrap_value=torch.tensor(2.0)
    )
    assert advantages.item() == pytest.approx(2.3)
    assert returns.item() == pytest.approx(2.8)


def test_ppo_loss_is_finite_and_differentiable():
    logits = torch.tensor([0.0, 0.1], requires_grad=True)
    loss = clipped_policy_loss(logits, torch.zeros(2), torch.tensor([1.0, -1.0]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_normalize_constant_advantages():
    result = normalize_advantages(torch.ones(4))
    assert torch.isfinite(result).all()
    assert torch.allclose(result, torch.zeros(4))


def test_action_mask_blocks_invalid_actions():
    model = MultiAgentActorCritic(["blue_0"], 3, 4, 3)
    obs = torch.zeros(3)
    mask = torch.tensor([False, True, False, False])
    for _ in range(10):
        action, _, _ = model.act("blue_0", obs, mask)
        assert action.item() == 1
    with pytest.raises(ValueError):
        model.act("blue_0", obs, torch.zeros(4, dtype=torch.bool))
