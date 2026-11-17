import numpy as np
from autocd.baselines.cc4_heuristic import CC4AlertGuidedPolicy


class FakeWrapper:
    def subnets(self, agent):
        return ["zone_a"]

    def hosts(self, agent):
        return [f"zone_a_user_host_{i}" for i in range(16)]

    def action_labels(self, agent):
        return ["Sleep"] + [f"Analyse {h}" for h in self.hosts(agent)]


def test_alert_guided_policy_analyses_flagged_host():
    wrapper = FakeWrapper()
    obs = np.zeros(200, dtype=np.float32)
    # Layout: mission + subnet features (27) + process[16] + connection[16].
    obs[1 + 27 + 4] = 1.0
    mask = np.ones(len(wrapper.action_labels("blue_agent_0")), dtype=bool)
    action = CC4AlertGuidedPolicy().act(wrapper, "blue_agent_0", obs, mask)
    assert wrapper.action_labels("blue_agent_0")[action] == "Analyse zone_a_user_host_4"


def test_alert_guided_policy_uses_sleep_without_alerts():
    wrapper = FakeWrapper()
    obs = np.zeros(200, dtype=np.float32)
    mask = np.ones(len(wrapper.action_labels("blue_agent_0")), dtype=bool)
    action = CC4AlertGuidedPolicy().act(wrapper, "blue_agent_0", obs, mask)
    assert action == 0
