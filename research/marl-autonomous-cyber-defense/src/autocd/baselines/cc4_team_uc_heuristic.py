"""Stateful Team-UC-inspired heuristic baseline for CAGE Challenge 4.

This is an independent reimplementation inspired by the published description,
not the original competition submission. It omits file flags unavailable in
BlueFlatWrapper's standard vector and should be reported with that limitation.
"""
from __future__ import annotations
import numpy as np

SUBNET_ORDER = tuple(sorted((
    "restricted_zone_a_subnet", "operational_zone_a_subnet",
    "restricted_zone_b_subnet", "operational_zone_b_subnet",
    "contractor_network_subnet", "public_access_zone_subnet",
    "admin_network_subnet", "office_network_subnet", "internet_subnet",
)))


class CC4TeamUCInspiredPolicy:
    SUBNET_BLOCK = 59
    HOSTS_PER_SUBNET = 16
    SUBNET_FEATURES = 27

    def __init__(self):
        self.steps = {}
        self.last_analyzed = {}
        self.removed = {}
        self.deployed_decoys = {}

    @staticmethod
    def _find(labels, mask, predicate):
        for index, label in enumerate(labels):
            if mask[index] and predicate(str(label)):
                return index
        return None

    def _traffic_action(self, labels, mask, subnet, blocked, desired_block):
        for target_index, target in enumerate(SUBNET_ORDER):
            if target == subnet or bool(blocked[target_index]) == bool(desired_block[target_index]):
                continue
            command = "BlockTrafficZone" if desired_block[target_index] else "AllowTrafficZone"
            index = self._find(
                labels, mask,
                lambda label: label.startswith(command + " ")
                and f"{target} (" in label
                and f"<- {subnet} (" in label,
            )
            if index is not None:
                return index
        return None

    def _reactive_action(self, agent, labels, mask, process_hosts, connection_hosts, step):
        for host in sorted(connection_hosts):
            index = self._find(labels, mask, lambda label: label == f"Restore {host}")
            if index is not None:
                self.last_analyzed[agent][host] = step
                return index
        for host in sorted(process_hosts):
            index = self._find(labels, mask, lambda label: label == f"Remove {host}")
            if index is None:
                index = self._find(labels, mask, lambda label: label == f"Restore {host}")
            if index is not None:
                self.removed[agent].add(host)
                self.last_analyzed[agent][host] = step
                return index
        return None

    def _proactive_action(self, agent, mission_phase, labels, mask, step):
        if mission_phase in (1, 2):
            monitor = self._find(labels, mask, lambda label: label == "Monitor")
            if monitor is not None:
                return monitor
        candidates = []
        for index, label in enumerate(labels):
            prefix = "Analyse "
            if mask[index] and str(label).startswith(prefix):
                host = str(label)[len(prefix):]
                if host not in self.removed[agent]:
                    candidates.append((self.last_analyzed[agent].get(host, -1), host, index))
        if candidates:
            _, host, index = min(candidates)
            self.last_analyzed[agent][host] = step
            return index
        for index, label in enumerate(labels):
            prefix = "DeployDecoy "
            if mask[index] and str(label).startswith(prefix):
                host = str(label)[len(prefix):]
                if host not in self.deployed_decoys[agent]:
                    self.deployed_decoys[agent].add(host)
                    return index
        return self._find(labels, mask, lambda label: label == "Monitor")

    def act(self, wrapper, agent, observation, action_mask):
        obs = np.asarray(observation).reshape(-1)
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        labels = wrapper.action_labels(agent)
        if mask.size != len(labels) or not mask.any():
            raise ValueError(f"Invalid action mask or labels for {agent}")
        self.steps[agent] = self.steps.get(agent, 0) + 1
        step = self.steps[agent]
        self.last_analyzed.setdefault(agent, {})
        self.removed.setdefault(agent, set())
        self.deployed_decoys.setdefault(agent, set())
        mission_phase = int(obs[0])

        process_hosts, connection_hosts = set(), set()
        hosts = wrapper.hosts(agent)
        for subnet_index, subnet in enumerate(wrapper.subnets(agent)):
            start = 1 + subnet_index * self.SUBNET_BLOCK
            process_start = start + self.SUBNET_FEATURES
            connection_start = process_start + self.HOSTS_PER_SUBNET
            if connection_start + self.HOSTS_PER_SUBNET > obs.size:
                raise ValueError("Observation too short for CC4 BlueFlatWrapper layout")
            local_hosts = [h for h in hosts if subnet in h and "router" not in h]
            for host_index, host in enumerate(local_hosts):
                if bool(obs[process_start + host_index]):
                    process_hosts.add(host)
                if bool(obs[connection_start + host_index]):
                    connection_hosts.add(host)

        action = self._reactive_action(
            agent, labels, mask, process_hosts, connection_hosts, step
        )
        if action is not None:
            return int(action)

        for subnet_index, subnet in enumerate(wrapper.subnets(agent)):
            start = 1 + subnet_index * self.SUBNET_BLOCK
            blocked = obs[start + 9:start + 18]
            desired_block = obs[start + 18:start + 27]
            action = self._traffic_action(labels, mask, subnet, blocked, desired_block)
            if action is not None:
                return int(action)

        action = self._proactive_action(agent, mission_phase, labels, mask, step)
        if action is not None:
            return int(action)
        sleep = self._find(labels, mask, lambda label: label.lower() == "sleep")
        return int(sleep if sleep is not None else np.flatnonzero(mask)[0])
