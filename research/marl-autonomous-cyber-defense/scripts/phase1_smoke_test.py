from autocd.envs.mock_enterprise import MockEnterpriseEnv
from autocd.risk import RiskScorer
from autocd.coordination import RiskAwareCoordinator
import numpy as np

def main():
    env = MockEnterpriseEnv(seed=42)
    obs, _ = env.reset(seed=42)
    assert set(obs) == set(env.possible_agents)
    assert all(x.shape == (env.observation_dim,) for x in obs.values())
    risk = RiskScorer().assess(.8, .9, .7, 1.)
    tasks = RiskAwareCoordinator().assign({"host-01": risk.score})
    actions = {a: 0 for a in env.possible_agents}
    nxt, rewards, *_ = env.step(actions)
    assert set(nxt) == set(env.possible_agents)
    assert all(np.isfinite(list(rewards.values())))
    assert tasks
    print("PHASE 1 MOCK SMOKE TEST: PASS")
    print(f"risk={risk.score:.3f} ({risk.severity}); assignments={tasks}")
    print("This validates the mock interface only, not CybORG integration.")

if __name__ == "__main__": main()
