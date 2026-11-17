import pytest
import torch
from autocd.algorithms.hierarchical import RiskAwareHierarchicalActorCritic


@pytest.fixture
def policy():
    return RiskAwareHierarchicalActorCritic(
        ["blue_agent_0", "blue_agent_1"], 12, 7, 24, num_tasks=4, hidden_dim=32
    )


def test_samples_masked_tasks_and_actions(policy):
    obs = torch.randn(5, 12)
    risk = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])
    tm = torch.ones(5, 4, dtype=torch.bool)
    tm[:, 3] = False
    am = torch.ones(5, 7, dtype=torch.bool)
    am[:, 6] = False
    result = policy.act("blue_agent_0", obs, risk, tm, am)
    assert result["task"].shape == (5,)
    assert result["action"].shape == (5,)
    assert torch.all(result["task"] < 3)
    assert torch.all(result["action"] < 6)
    assert torch.isfinite(result["manager_log_prob"]).all()


def test_evaluate_reproduces_deterministic_log_probs(policy):
    obs, risk = torch.randn(3, 12), torch.tensor([0.1, 0.4, 0.9])
    sampled = policy.act("blue_agent_1", obs, risk, deterministic=True)
    evaluated = policy.evaluate_actions("blue_agent_1", obs, risk, sampled["task"], sampled["action"])
    torch.testing.assert_close(evaluated["manager_log_prob"], sampled["manager_log_prob"])
    torch.testing.assert_close(evaluated["worker_log_prob"], sampled["worker_log_prob"])


def test_rejects_invalid_risk_and_empty_mask(policy):
    with pytest.raises(ValueError, match="risk"):
        policy.act("blue_agent_0", torch.randn(1, 12), torch.tensor([1.2]))
    with pytest.raises(ValueError, match="Mask"):
        policy.act("blue_agent_0", torch.randn(1, 12), torch.tensor([0.5]),
                   task_mask=torch.zeros(1, 4, dtype=torch.bool))


def test_critic_shape(policy):
    assert policy.value(torch.randn(6, 24)).shape == (6,)
