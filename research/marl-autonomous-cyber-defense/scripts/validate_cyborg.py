"""Validate the pinned official CC4 runtime and the research adapter."""
import argparse
import json
import numpy as np

from autocd.envs.cyborg_adapter import CybORGAdapter
from autocd.baselines.cc4_heuristic import CC4AlertGuidedPolicy


def _snapshot(seed):
    env = CybORGAdapter(seed=seed)
    try:
        observations, infos = env.reset(seed=seed)
        masks = env.action_masks()
        return observations, masks, env.observation_dims.copy(), env.action_dims.copy()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--one-step", action="store_true")
    parser.add_argument("--verify-seed", action="store_true")
    args = parser.parse_args()

    from CybORG import CYBORG_VERSION
    if str(CYBORG_VERSION) != "4.0":
        raise RuntimeError(f"Expected CybORG 4.0, found {CYBORG_VERSION}")

    env = CybORGAdapter(seed=args.seed)
    try:
        obs, infos = env.reset(seed=args.seed)
        if set(obs) != set(env.EXPECTED_AGENTS):
            raise RuntimeError(f"Expected five Blue agents, got {sorted(obs)}")
        dims = set(env.observation_dims.values())
        action_dims = set(env.action_dims.values())
        if len(dims) != 1 or len(action_dims) != 1:
            raise RuntimeError("pad_spaces=True did not produce common observation/action dimensions")
        masks = env.action_masks()
        for agent, mask in masks.items():
            if mask.size != env.action_dims[agent] or not mask.any():
                raise RuntimeError(f"Invalid action mask for {agent}")
            labels = env.env.action_labels(agent)
            if len(labels) != mask.size:
                raise RuntimeError(f"Action labels/mask mismatch for {agent}")

        report = {
            "cyborg_version": str(CYBORG_VERSION),
            "agents": sorted(obs),
            "observation_dims": env.observation_dims,
            "action_dims": env.action_dims,
            "valid_action_counts": {a: int(m.sum()) for a, m in masks.items()},
            "seed": args.seed,
        }

        if args.verify_seed:
            obs2, masks2, obs_dims2, action_dims2 = _snapshot(args.seed)
            replay = all(np.array_equal(obs[a], obs2[a]) for a in env.EXPECTED_AGENTS)
            replay = replay and all(np.array_equal(masks[a], masks2[a]) for a in env.EXPECTED_AGENTS)
            replay = replay and obs_dims2 == env.observation_dims and action_dims2 == env.action_dims
            report["same_seed_initial_state_reproducible"] = bool(replay)
            if not replay:
                raise RuntimeError("Same-seed initial observations/action masks are not reproducible")

        if args.one_step:
            policy = CC4AlertGuidedPolicy()
            actions, selected_labels = {}, {}
            for agent in env.agents:
                action = policy.act(env.env, agent, obs[agent], masks[agent])
                if not masks[agent][action]:
                    raise RuntimeError(f"Heuristic selected a masked action for {agent}")
                actions[agent] = int(action)
                selected_labels[agent] = env.env.action_labels(agent)[action]
            _, rewards, terminated, truncated, _ = env.step(actions)
            report["one_step"] = {
                "selected_actions": selected_labels,
                "reward_agents": sorted(rewards),
                "terminated": terminated,
                "truncated": truncated,
                "all_rewards_finite": bool(np.isfinite(list(rewards.values())).all()),
            }
            if not report["one_step"]["all_rewards_finite"]:
                raise RuntimeError("Non-finite reward observed")

        print(json.dumps(report, indent=2, default=lambda x: x.tolist()
                         if isinstance(x, np.ndarray) else str(x)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
