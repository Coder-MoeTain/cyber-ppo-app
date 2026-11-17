"""Transparent alert-guided CC4 heuristic baseline (not an exact contest reproduction)."""
from __future__ import annotations
import numpy as np


class CC4AlertGuidedPolicy:
    """Analyse the most suspicious locally observed host; otherwise Sleep.

    BlueFlatWrapper encodes process-creation and network-connection event bits
    per local subnet. These are alert proxies, not ground-truth compromise labels.
    """
    SUBNET_BLOCK = 59
    SUBNET_FEATURES = 27
    MAX_HOSTS_PER_SUBNET = 16

    def act(self, wrapper, agent, observation, action_mask):
        obs = np.asarray(observation).reshape(-1)
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        labels = wrapper.action_labels(agent)
        if mask.size != len(labels):
            raise ValueError("action mask and action labels differ in length")
        if not mask.any():
            raise ValueError(f"No valid action for {agent}")

        host_scores = {}
        host_list = wrapper.hosts(agent)
        for subnet_index, subnet in enumerate(wrapper.subnets(agent)):
            local_hosts = [
                host for host in host_list
                if subnet in host and "router" not in host
            ]
            start = 1 + subnet_index * self.SUBNET_BLOCK
            process_start = start + self.SUBNET_FEATURES
            connection_start = process_start + self.MAX_HOSTS_PER_SUBNET
            if connection_start + self.MAX_HOSTS_PER_SUBNET > obs.size:
                raise ValueError("Observation too short for documented CC4 feature layout")
            for host_index, host in enumerate(local_hosts):
                process = float(obs[process_start + host_index])
                connection = float(obs[connection_start + host_index])
                score = process + connection
                if score > 0:
                    host_scores[host] = max(host_scores.get(host, 0.0), score)

        for host, _ in sorted(host_scores.items(), key=lambda item: (-item[1], item[0])):
            target = f"Analyse {host}"
            for idx, label in enumerate(labels):
                if mask[idx] and str(label) == target:
                    return int(idx)

        sleep = next((i for i, label in enumerate(labels)
                      if mask[i] and str(label).lower() == "sleep"), None)
        if sleep is not None:
            return int(sleep)
        return int(np.flatnonzero(mask)[0])
