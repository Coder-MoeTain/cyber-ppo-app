import numpy as np
from autocd.envs.cc4_tasks import CC4TaskActionMapper


class FakeWrapper:
    def action_labels(self, agent):
        return [
            "Sleep", "Monitor", "Analyse host_a", "Remove host_a",
            "Restore host_a", "BlockTrafficZone zone_a", "AllowTrafficZone zone_a",
            "DeployDecoy host_a", "[Invalid] Analyse host_b",
        ]


def test_cc4_task_mapper_creates_semantic_valid_action_masks():
    mapper = CC4TaskActionMapper(FakeWrapper())
    base = np.array([True, True, True, True, True, True, True, True, False])
    masks = mapper("blue_agent_0", np.zeros(5), {}, base)
    assert masks.shape == (6, 9)
    assert masks[0].tolist() == [True, True, False, False, False, False, False, False, False]
    assert masks[1][2]
    assert masks[2][3]
    assert masks[3][4] and masks[3][6]
    assert masks[4][5]
    assert masks[5][7]
    assert not masks[:, 8].any()


def test_unavailable_task_is_masked_not_selected():
    mapper = CC4TaskActionMapper(FakeWrapper())
    base = np.array([True, True, False, False, False, False, False, False, False])
    masks = mapper("blue_agent_0", np.zeros(5), {}, base)
    assert masks[0].any()
    assert not masks[1:].any()
