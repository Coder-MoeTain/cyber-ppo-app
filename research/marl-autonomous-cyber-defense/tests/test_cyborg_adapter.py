import numpy as np
from autocd.envs.cyborg_adapter import CybORGAdapter


class FakeBase:
    def __init__(self, scenario_generator):
        self.scenario_generator = scenario_generator


class Space:
    n = 3


class FakeWrapper:
    def __init__(self, env, pad_spaces=True):
        self.names = [f"blue_agent_{i}" for i in range(5)]
        self.last_seed = None

    def reset(self, seed=None):
        self.last_seed = seed
        return (
            {a: np.zeros(4, dtype=np.float32) for a in self.names},
            {a: {} for a in self.names},
        )

    def action_space(self, agent):
        return Space()

    def action_mask(self, agent):
        return [True, False, True]

    def action_labels(self, agent):
        return ["Sleep", "Invalid", "Analyse"]

    def step(self, actions):
        assert all(v in (0, 2) for v in actions.values())
        return (
            {a: np.ones(4, dtype=np.float32) for a in self.names},
            {a: 1.0 for a in self.names},
            {a: False for a in self.names},
            {a: False for a in self.names},
            {a: {} for a in self.names},
        )

    def close(self):
        pass


def test_adapter_contract_fake_backend_and_seed_forwarding():
    env = CybORGAdapter(
        scenario_generator=object(), seed=11,
        cyborg_factory=FakeBase, wrapper_factory=FakeWrapper,
    )
    obs, info = env.reset()
    assert len(obs) == 5
    assert env.env.last_seed == 11
    assert env.observation_dims["blue_agent_0"] == 4
    assert env.action_dims["blue_agent_0"] == 3
    assert info["blue_agent_0"]["action_mask"].tolist() == [True, False, True]
    out = env.step({a: 0 for a in env.agents})
    assert out[1]["blue_agent_0"] == 1.0
    env.close()


def test_explicit_reset_seed_overrides_constructor_seed():
    env = CybORGAdapter(
        scenario_generator=object(), seed=11,
        cyborg_factory=FakeBase, wrapper_factory=FakeWrapper,
    )
    env.reset(seed=99)
    assert env.env.last_seed == 99
    env.close()
