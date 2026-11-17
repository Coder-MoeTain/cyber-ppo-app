import numpy as np

from autocd.baselines.cc4_team_uc_heuristic import (
    CC4TeamUCInspiredPolicy, SUBNET_ORDER,
)


SOURCE = "restricted_zone_a_subnet"
HOSTS = [f"{SOURCE}_user_host_{i}" for i in range(16)]
TARGET = "operational_zone_a_subnet"


class FakeWrapper:
    def subnets(self, agent):
        return [SOURCE]

    def hosts(self, agent):
        return HOSTS

    def action_labels(self, agent):
        return (
            ["Sleep", "Monitor"]
            + [f"Analyse {h}" for h in HOSTS]
            + [f"Remove {h}" for h in HOSTS]
            + [f"Restore {h}" for h in HOSTS]
            + [f"BlockTrafficZone {dst} (10.0.0.0/24) <- {SOURCE} (10.0.1.0/24)"
               for dst in SUBNET_ORDER if dst != SOURCE]
            + [f"AllowTrafficZone {dst} (10.0.0.0/24) <- {SOURCE} (10.0.1.0/24)"
               for dst in SUBNET_ORDER if dst != SOURCE]
            + [f"DeployDecoy {h}" for h in HOSTS]
        )


def setup():
    wrapper = FakeWrapper()
    obs = np.zeros(210, dtype=np.float32)
    labels = wrapper.action_labels("blue_agent_0")
    mask = np.ones(len(labels), dtype=bool)
    return wrapper, obs, labels, mask


def test_process_event_triggers_remove():
    wrapper, obs, labels, mask = setup()
    obs[1 + 27] = 1.0
    action = CC4TeamUCInspiredPolicy().act(wrapper, "blue_agent_0", obs, mask)
    assert labels[action] == f"Remove {HOSTS[0]}"


def test_connection_event_triggers_restore():
    wrapper, obs, labels, mask = setup()
    obs[1 + 27 + 16] = 1.0
    action = CC4TeamUCInspiredPolicy().act(wrapper, "blue_agent_0", obs, mask)
    assert labels[action] == f"Restore {HOSTS[0]}"


def test_communication_policy_mismatch_triggers_block():
    wrapper, obs, labels, mask = setup()
    start = 1
    target_idx = SUBNET_ORDER.index(TARGET)
    obs[start + 18 + target_idx] = 1.0
    action = CC4TeamUCInspiredPolicy().act(wrapper, "blue_agent_0", obs, mask)
    assert labels[action].startswith(f"BlockTrafficZone {TARGET} ")


def test_active_mission_phase_monitors_and_preplanning_analyzes():
    wrapper, obs, labels, mask = setup()
    policy = CC4TeamUCInspiredPolicy()
    action = policy.act(wrapper, "blue_agent_0", obs, mask)
    assert labels[action] == f"Analyse {HOSTS[0]}"

    obs[0] = 1.0
    action = policy.act(wrapper, "blue_agent_0", obs, mask)
    assert labels[action] == "Monitor"
