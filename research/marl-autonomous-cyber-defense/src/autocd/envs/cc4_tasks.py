"""Semantic manager-task to CC4 action-mask mapping."""
from __future__ import annotations
import numpy as np


class CC4TaskActionMapper:
    """Restrict each high-level task to its operational CC4 action family.

    Task order is part of the checkpoint/config contract:
    0 monitor, 1 investigate, 2 remediate, 3 recover, 4 isolate, 5 deceive.
    """
    TASKS = ("monitor", "investigate", "remediate", "recover", "isolate", "deceive")
    PREFIXES = {
        "monitor": ("Monitor", "Sleep"),
        "investigate": ("Analyse",),
        "remediate": ("Remove", "RemoveOtherSessions"),
        "recover": ("Restore", "AllowTrafficZone"),
        "isolate": ("BlockTrafficZone",),
        "deceive": ("DeployDecoy",),
    }

    def __init__(self, wrapper):
        self.wrapper = wrapper

    def __call__(self, agent, observation, info, action_mask):
        base_mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        labels = self.wrapper.action_labels(agent)
        if len(labels) != base_mask.size:
            raise ValueError(f"Action labels/mask length mismatch for {agent}")
        result = np.zeros((len(self.TASKS), base_mask.size), dtype=bool)
        for task_id, task in enumerate(self.TASKS):
            prefixes = self.PREFIXES[task]
            result[task_id] = np.asarray([
                bool(base_mask[i]) and any(str(label).startswith(prefix) for prefix in prefixes)
                for i, label in enumerate(labels)
            ], dtype=bool)
        if not result.any(axis=1)[0]:
            raise RuntimeError("CC4 monitor task must always expose Monitor or Sleep")
        return result
