"""Collect local CC4 observations with offline red-session labels.

Privileged simulator state is read only to produce supervised training labels.
It is never passed to the policy or risk predictor at inference time.
"""
import argparse
import numpy as np

from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from autocd.envs.cyborg_adapter import CybORGAdapter


def red_present_in_agent_zone(adapter, agent):
    base_env = adapter.env.env
    state = base_env.environment_controller.state
    for hostname in adapter.env.hosts(agent):
        host = state.hosts.get(hostname)
        if host is None:
            continue
        sessions = getattr(host, "sessions", {})
        for session_agent, session_list in sessions.items():
            if "red" in str(session_agent).lower() and len(session_list) > 0:
                return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")

    X, agent_ids, labels, episode_ids = [], [], [], []
    agent_names = [f"blue_agent_{i}" for i in range(5)]
    for seed in args.seeds:
        generator = EnterpriseScenarioGenerator(steps=args.steps)
        env = CybORGAdapter(scenario_generator=generator, seed=seed)
        try:
            observations, _ = env.reset(seed=seed)
            for _ in range(args.steps):
                for idx, agent in enumerate(agent_names):
                    if agent not in observations:
                        continue
                    X.append(np.asarray(observations[agent], dtype=np.float32))
                    agent_ids.append(idx)
                    labels.append(red_present_in_agent_zone(env, agent))
                    episode_ids.append(seed)

                actions = {}
                for agent in env.agents:
                    labels_for_agent = env.env.action_labels(agent)
                    sleep_idx = next(i for i, label in enumerate(labels_for_agent)
                                     if str(label).lower() == "sleep")
                    if not env.action_masks()[agent][sleep_idx]:
                        raise RuntimeError(f"Sleep unexpectedly masked for {agent}")
                    actions[agent] = int(sleep_idx)

                observations, _, terminated, truncated, _ = env.step(actions)
                if all(bool(terminated.get(a, False) or truncated.get(a, False))
                       for a in agent_names):
                    break
        finally:
            env.close()

    if not X:
        raise RuntimeError("No CC4 samples collected")
    np.savez_compressed(
        args.output,
        observations=np.stack(X),
        agent_ids=np.asarray(agent_ids, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        agent_names=np.asarray(agent_names),
        seeds=np.asarray(args.seeds, dtype=np.int64),
    )
    print({"output": args.output, "samples": len(X),
           "positive_rate": float(np.mean(labels)), "episodes": len(args.seeds)})


if __name__ == "__main__":
    main()
