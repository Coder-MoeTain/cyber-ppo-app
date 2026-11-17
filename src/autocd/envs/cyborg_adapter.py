"""CybORG CAGE Challenge 4 adapter based on the official BlueFlatWrapper API.

The adapter intentionally keeps CybORG optional at import time so unit tests and
mock experiments can run without installing the simulator.
"""
from __future__ import annotations
from typing import Any
import numpy as np


class CybORGAdapter:
    """Expose the five CC4 Blue agents through a small parallel-style interface."""
    EXPECTED_AGENTS = tuple(f"blue_agent_{i}" for i in range(5))

    def __init__(self, scenario_generator=None, *, pad_spaces: bool = True,
                 seed: int | None = None, cyborg_factory=None, wrapper_factory=None):
        self.seed = seed
        if cyborg_factory is None or wrapper_factory is None:
            try:
                from CybORG import CybORG
                from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
                from CybORG.Agents.Wrappers import BlueFlatWrapper
            except ImportError as exc:
                raise RuntimeError(
                    "CybORG is not installed. Install the CAGE Challenge 4 CybORG "
                    "release and record its exact commit."
                ) from exc
            cyborg_factory = cyborg_factory or CybORG
            wrapper_factory = wrapper_factory or BlueFlatWrapper
            scenario_generator = scenario_generator or EnterpriseScenarioGenerator()
        if scenario_generator is None:
            raise ValueError("scenario_generator is required with injected factories")
        base_env = cyborg_factory(scenario_generator=scenario_generator)
        self.env = wrapper_factory(env=base_env, pad_spaces=pad_spaces)
        self.possible_agents = list(self.EXPECTED_AGENTS)
        self.agents = self.possible_agents.copy()
        self.observation_dims: dict[str, int] = {}
        self.action_dims: dict[str, int] = {}
        self._last_info: dict[str, dict] = {}
        self._closed = False

    @staticmethod
    def _vector(value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if not np.isfinite(arr).all():
            raise ValueError("CybORG observation contains NaN or infinity")
        return arr

    def _observations(self, observations):
        out = {}
        for agent in self.agents:
            if agent not in observations:
                continue
            out[agent] = self._vector(observations[agent])
            self.observation_dims.setdefault(agent, out[agent].size)
            if self.observation_dims[agent] != out[agent].size:
                raise ValueError(f"Observation dimension changed for {agent}")
        return out

    def action_masks(self):
        masks = {}
        for agent in self.agents:
            raw = self.env.action_mask(agent)
            mask = np.asarray(raw, dtype=bool).reshape(-1)
            if mask.size != self.action_dims[agent]:
                raise ValueError(f"Action mask dimension mismatch for {agent}")
            if not mask.any():
                labels = self.env.action_labels(agent)
                sleep_idx = next((i for i, x in enumerate(labels)
                                  if "sleep" in str(x).lower()), None)
                if sleep_idx is None:
                    raise ValueError(f"No valid actions and no Sleep action for {agent}")
                mask[sleep_idx] = True
            masks[agent] = mask
        return masks

    def reset(self, seed: int | None = None):
        if self._closed:
            raise RuntimeError("Environment is closed")
        obs, info = self.env.reset()
        self.agents = self.possible_agents.copy()
        for agent in self.agents:
            self.action_dims[agent] = int(self.env.action_space(agent).n)
        vectors = self._observations(obs)
        self._last_info = info or {}
        infos = {a: dict(self._last_info.get(a, {})) for a in vectors}
        for a, mask in self.action_masks().items():
            infos.setdefault(a, {})["action_mask"] = mask
        return vectors, infos

    def step(self, actions: dict[str, int]):
        if set(actions) != set(self.agents):
            raise ValueError("Actions must be supplied for every active Blue agent")
        for agent, action in actions.items():
            if not isinstance(action, (int, np.integer)):
                raise TypeError(f"Action for {agent} must be an integer index")
            if not 0 <= int(action) < self.action_dims[agent]:
                raise ValueError(f"Action index out of range for {agent}")
        obs, rewards, terminated, truncated, infos = self.env.step(
            {a: int(v) for a, v in actions.items()}
        )
        done = {a: bool(terminated.get(a, False) or truncated.get(a, False))
                for a in self.agents}
        vectors = self._observations(obs)
        result_infos = {a: dict(infos.get(a, {})) for a in self.agents}
        for a in self.agents:
            result_infos.setdefault(a, {})["action_mask"] = np.asarray(
                self.env.action_mask(a), dtype=bool
            )
        if all(done.values()):
            self.agents = []
        return (
            vectors,
            {a: float(rewards.get(a, 0.0)) for a in done},
            {a: bool(terminated.get(a, False)) for a in done},
            {a: bool(truncated.get(a, False)) for a in done},
            result_infos,
        )

    def close(self):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
        self.agents = []
        self._closed = True
