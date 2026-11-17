"""Evaluate Independent PPO or the alert-guided heuristic on held-out CC4 seeds."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from autocd.algorithms.independent_ppo import IndependentActorCritic
from autocd.baselines.cc4_heuristic import CC4AlertGuidedPolicy
from autocd.envs.cyborg_adapter import CybORGAdapter


def red_host_count(adapter, hostnames):
    state = adapter.env.env.environment_controller.state
    return sum(
        1 for hostname in hostnames
        if hostname in state.hosts and any(
            "red" in str(agent).lower() and len(sessions) > 0
            for agent, sessions in getattr(state.hosts[hostname], "sessions", {}).items()
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", choices=["ippo", "heuristic"], required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--episode-steps", type=int, default=500)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    checkpoint = None
    if args.baseline == "ippo":
        if not args.checkpoint:
            p.error("--checkpoint is required for IPPO")
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    env = CybORGAdapter(
        scenario_generator=EnterpriseScenarioGenerator(steps=args.episode_steps)
    )
    records = []
    try:
        if checkpoint is not None:
            env.reset(seed=args.seeds[0])
            if list(env.possible_agents) != checkpoint["agents"]:
                raise RuntimeError("Checkpoint agent order differs from CC4")
            model = IndependentActorCritic(
                checkpoint["observation_dims"], checkpoint["action_dims"],
                checkpoint["hidden_dim"],
            ).to(args.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
        else:
            policy = CC4AlertGuidedPolicy()

        for seed in args.seeds:
            observations, infos = env.reset(seed=seed)
            totals = {a: 0.0 for a in env.possible_agents}
            actions_count = Counter()
            invalid, total, steps = 0, 0, 0
            for _ in range(args.episode_steps):
                masks = env.action_masks()
                actions = {}
                for agent in env.agents:
                    mask = masks[agent]
                    if checkpoint is not None:
                        obs = torch.as_tensor(observations[agent], dtype=torch.float32,
                                               device=args.device)
                        with torch.no_grad():
                            dist = model.distribution(
                                agent, obs,
                                torch.as_tensor(mask, dtype=torch.bool, device=args.device),
                            )
                            action = int(dist.probs.argmax(-1).item())
                    else:
                        action = policy.act(env.env, agent, observations[agent], mask)
                    actions[agent] = action
                    total += 1
                    invalid += int(not bool(mask[action]))
                    label = env.env.action_labels(agent)[action]
                    actions_count[str(label).split()[0]] += 1

                observations, rewards, terms, truncs, infos = env.step(actions)
                for agent, reward in rewards.items():
                    totals[agent] += float(reward)
                steps += 1
                if all(bool(terms.get(a, False) or truncs.get(a, False))
                       for a in env.possible_agents):
                    break

            records.append({
                "seed": int(seed), "steps": steps,
                "team_reward_sum": float(sum(totals.values())),
                "per_agent_reward": totals,
                "red_session_hosts_by_agent": {
                    a: red_host_count(env, env.env.hosts(a)) for a in env.possible_agents
                },
                "invalid_action_rate": float(invalid / max(total, 1)),
                "action_counts": dict(actions_count),
            })
    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    summary = {
        "baseline": args.baseline,
        "episodes": len(records),
        "mean_team_reward_sum": float(np.mean([r["team_reward_sum"] for r in records])),
        "std_team_reward_sum": float(np.std([r["team_reward_sum"] for r in records], ddof=1))
        if len(records) > 1 else 0.0,
        "mean_invalid_action_rate": float(np.mean([r["invalid_action_rate"] for r in records])),
        "episode_records": str(output),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
