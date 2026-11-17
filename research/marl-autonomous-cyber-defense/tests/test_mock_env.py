from autocd.envs.mock_enterprise import MockEnterpriseEnv

def test_reset_step():
    env = MockEnterpriseEnv(seed=7)
    obs, _ = env.reset(seed=7)
    acts = {a: 0 for a in env.possible_agents}
    nxt, rewards, terms, truncs, infos = env.step(acts)
    assert len(obs) == len(nxt) == len(rewards) == 3
    assert all(isinstance(v, float) for v in rewards.values())
