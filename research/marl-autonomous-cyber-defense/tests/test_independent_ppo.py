import numpy as np
from autocd.algorithms.independent_ppo import IPPOConfig, IndependentPPOTrainer
from autocd.envs.mock_enterprise import MockEnterpriseEnv


def test_ippo_mock_training_and_masking():
    class MaskedEnv(MockEnterpriseEnv):
        def action_masks(self):
            return {a: np.array([True, False, True, False, False])
                    for a in self.possible_agents}
    env = MaskedEnv(n_hosts=5, episode_length=3, seed=8)
    trainer = IndependentPPOTrainer(
        env, IPPOConfig(update_epochs=1, minibatch_size=3, hidden_dim=32), seed=8
    )
    rollout = trainer.collect_rollout(3)
    assert all(all(action in (0, 2) for action in rollout["agents"][a]["actions"])
               for a in env.possible_agents)
    metrics = trainer.update(rollout)
    assert np.isfinite(metrics["mean_actor_loss"])
    assert set(metrics["agents"]) == set(env.possible_agents)


def test_ippo_training_callback_receives_progress():
    trainer = IndependentPPOTrainer(
        MockEnterpriseEnv(n_hosts=5, episode_length=3, seed=61),
        IPPOConfig(update_epochs=1, minibatch_size=3, hidden_dim=32), seed=61,
    )
    updates = []
    trainer.train(7, 3, on_update=updates.append)
    assert [row["environment_steps"] for row in updates] == [3, 6, 7]
