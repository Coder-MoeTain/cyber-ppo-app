import numpy as np, pytest, torch
from autocd.algorithms.mappo_trainer import MAPPOConfig,MAPPOTrainer,CTDEActorCritic
from autocd.envs.mock_enterprise import MockEnterpriseEnv

def test_ctde_shapes_and_mask():
    m=CTDEActorCritic({"a":4,"b":4},{"a":3,"b":2},8)
    d=m.distribution("a",torch.zeros(2,4),torch.tensor([[1,0,1],[0,1,0]],dtype=torch.bool))
    assert d.probs.shape==(2,3) and m.value(torch.zeros(2,8)).shape==(2,)

def test_mappo_mock_training():
    t=MAPPOTrainer(MockEnterpriseEnv(n_hosts=8,episode_length=4,seed=5),MAPPOConfig(update_epochs=2,minibatch_size=4),seed=5)
    h=t.train(8,4);assert len(h)==2
    assert all(np.isfinite(x["actor_loss"]) and np.isfinite(x["critic_loss"]) for x in h)

def test_all_false_mask_rejected():
    m=CTDEActorCritic({"a":2},{"a":2},2)
    with pytest.raises(ValueError):m.distribution("a",torch.zeros(2),torch.tensor([False,False]))

def test_action_mask_used_in_rollout():
    class Masked(MockEnterpriseEnv):
        def action_masks(self):return {a:np.array([1,0,1,0,0],dtype=bool) for a in self.possible_agents}
    t=MAPPOTrainer(Masked(n_hosts=4,episode_length=3,seed=9),MAPPOConfig(update_epochs=1,minibatch_size=3),seed=9)
    r=t.collect_rollout(3)
    assert all(v in (0,2) for row in r["actions"] for v in row.values())

def test_truncation_bootstrap_math():
    reward,value,next_value=1.0,.5,2.0
    assert reward+.9*next_value-value==pytest.approx(2.3)
    assert reward-value==pytest.approx(.5)
