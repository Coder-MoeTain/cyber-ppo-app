import numpy as np
import pytest
import torch
from autocd.algorithms.hierarchical_trainer import (
    RAHMAPPOConfig, RiskAwareHierarchicalTrainer,
)
from autocd.envs.mock_enterprise import MockEnterpriseEnv


def mock_risk(agent, observation, info):
    # Mock-only ground-truth feature: weighted compromised-host exposure.
    return float(np.clip(observation[5], 0.0, 1.0))


def test_hierarchical_trainer_collects_and_updates():
    env = MockEnterpriseEnv(n_hosts=5, episode_length=4, seed=10)
    trainer = RiskAwareHierarchicalTrainer(
        env, mock_risk,
        RAHMAPPOConfig(update_epochs=1, minibatch_size=4, hidden_dim=32),
        seed=10,
    )
    rollout = trainer.collect_rollout(4)
    assert len(rollout["tasks"]) == 4
    assert set(rollout["tasks"][0]) == set(env.possible_agents)
    assert all(0.0 <= r <= 1.0 for row in rollout["risks"] for r in row.values())
    metrics = trainer.update(rollout)
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["manager_entropy"])
    assert np.isfinite(metrics["worker_entropy"])


def test_hierarchical_trainer_rejects_missing_risk_function():
    with pytest.raises(TypeError):
        RiskAwareHierarchicalTrainer(MockEnterpriseEnv(), None)


def test_hierarchical_trainer_rejects_bad_risk_value():
    env = MockEnterpriseEnv()
    trainer = RiskAwareHierarchicalTrainer(env, lambda *_: 2.0)
    with pytest.raises(ValueError, match="risk_fn"):
        trainer.collect_rollout(1)


def test_seeded_episode_resets_are_deterministic():
    env_a, env_b = MockEnterpriseEnv(episode_length=2), MockEnterpriseEnv(episode_length=2)
    a = RiskAwareHierarchicalTrainer(env_a, mock_risk, seed=7)
    b = RiskAwareHierarchicalTrainer(env_b, mock_risk, seed=7)
    ra, rb = a.collect_rollout(5), b.collect_rollout(5)
    assert ra["rewards"] == rb["rewards"]



def test_trainer_enforces_task_specific_action_masks():
    def mapper(agent, observation, info, base_mask):
        result = np.zeros((6, len(base_mask)), dtype=bool)
        result[0, 0] = bool(base_mask[0])
        result[1, 2] = bool(base_mask[2])
        return result

    env = MockEnterpriseEnv(n_hosts=5, episode_length=4, seed=12)
    trainer = RiskAwareHierarchicalTrainer(
        env, mock_risk,
        RAHMAPPOConfig(update_epochs=1, minibatch_size=4, hidden_dim=32),
        seed=12, task_action_mapper=mapper,
    )
    rollout = trainer.collect_rollout(4)
    for t, row in enumerate(rollout["tasks"]):
        for agent, task in row.items():
            action = rollout["actions"][t][agent]
            assert task in (0, 1)
            assert action == (0 if task == 0 else 2)
    metrics = trainer.update(rollout)
    assert np.isfinite(metrics["actor_loss"])


def test_risk_rule_manager_uses_threshold_and_freezes_manager_parameters():
    env = MockEnterpriseEnv(n_hosts=5, episode_length=4, seed=21)
    trainer = RiskAwareHierarchicalTrainer(
        env, lambda *_: 0.9,
        RAHMAPPOConfig(update_epochs=1, minibatch_size=4, hidden_dim=32),
        seed=21, manager_mode="risk_rule", risk_rule_threshold=0.6,
    )
    before = {
        key: value.detach().clone()
        for key, value in trainer.model.managers["blue_agent_0"].state_dict().items()
    }
    rollout = trainer.collect_rollout(4)
    assert all(row["blue_agent_0"] == 1 for row in rollout["tasks"])
    metrics = trainer.update(rollout)
    after = trainer.model.managers["blue_agent_0"].state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert metrics["manager_entropy"] == pytest.approx(0.0)


def test_risk_rule_manager_uses_monitor_task_for_low_risk():
    env = MockEnterpriseEnv(n_hosts=5, episode_length=4, seed=22)
    trainer = RiskAwareHierarchicalTrainer(
        env, lambda *_: 0.1,
        RAHMAPPOConfig(update_epochs=1, minibatch_size=4, hidden_dim=32),
        seed=22, manager_mode="risk_rule",
    )
    rollout = trainer.collect_rollout(4)
    assert all(row["blue_agent_0"] == 0 for row in rollout["tasks"])


def test_training_callback_receives_rollout_progress():
    env = MockEnterpriseEnv(n_hosts=5, episode_length=3, seed=31)
    trainer = RiskAwareHierarchicalTrainer(
        env, mock_risk,
        RAHMAPPOConfig(update_epochs=1, minibatch_size=3, hidden_dim=32),
        seed=31,
    )
    updates = []
    history = trainer.train(7, 3, on_update=updates.append)
    assert len(updates) == len(history) == 3
    assert [row["environment_steps"] for row in updates] == [3, 6, 7]
