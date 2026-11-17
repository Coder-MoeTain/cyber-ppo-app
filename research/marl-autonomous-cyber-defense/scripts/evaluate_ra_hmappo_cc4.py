"""Evaluate an RA-HMAPPO checkpoint on held-out CC4 seeds."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from autocd.algorithms.hierarchical import RiskAwareHierarchicalActorCritic
from autocd.envs.cyborg_adapter import CybORGAdapter
from autocd.risk_model import RiskPredictor


def red_host_count(adapter, hostnames):
    state = adapter.env.env.environment_controller.state
    count = 0
    for hostname in hostnames:
        host = state.hosts.get(hostname)
        if host is not None and any(
            "red" in str(agent).lower() and len(sessions) > 0
            for agent, sessions in getattr(host, "sessions", {}).items()
        ):
            count += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--risk-checkpoint", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--episode-steps", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    env = CybORGAdapter(
        scenario_generator=EnterpriseScenarioGenerator(steps=args.episode_steps)
    )
    records = []
    try:
        observations, _ = env.reset(seed=args.seeds[0])
        if list(env.possible_agents) != checkpoint["agents"]:
            raise RuntimeError("Checkpoint agent order does not match pinned CC4 wrapper")
        model = RiskAwareHierarchicalActorCritic(
            checkpoint["agents"], checkpoint["observation_dim"],
            checkpoint["action_dim"], checkpoint["global_state_dim"],
            checkpoint["config"]["num_tasks"], checkpoint["config"]["hidden_dim"],
        ).to(args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        risk_fn = RiskPredictor.load(args.risk_checkpoint, device=args.device)

        for seed in args.seeds:
            observations, infos = env.reset(seed=seed)
            totals = {a: 0.0 for a in env.possible_agents}
            action_counts = Counter()
            invalid_actions = 0
            total_actions = 0
            steps = 0
            for _ in range(args.episode_steps):
                masks = env.action_masks()
                actions = {}
                for agent in env.agents:
                    obs = np.asarray(observations[agent], dtype=np.float32)
                    risk = risk_fn(agent, obs, infos.get(agent, {}))
                    mask = masks[agent]
                    with torch.no_grad():
                        decision = model.act(
                            agent,
                            torch.as_tensor(obs, dtype=torch.float32, device=args.device),
                            torch.tensor([risk], dtype=torch.float32, device=args.device),
                            action_mask=torch.as_tensor(mask, dtype=torch.bool, device=args.device),
                            deterministic=True,
                        )
                    action = int(decision["action"].item())
                    actions[agent] = action
                    total_actions += 1
                    invalid_actions += int(not bool(mask[action]))
                    label = env.env.action_labels(agent)[action]
                    category = str(label).split()[0].replace("[Invalid]", "Invalid")
                    action_counts[category] += 1

                observations, rewards, terms, truncs, infos = env.step(actions)
                for agent, reward in rewards.items():
                    totals[agent] += float(reward)
                steps += 1
                if all(bool(terms.get(a, False) or truncs.get(a, False))
                       for a in env.possible_agents):
                    break

            host_counts = {
                a: red_host_count(env, env.env.hosts(a)) for a in env.possible_agents
            }
            records.append({
                "seed": int(seed),
                "steps": steps,
                "team_reward_sum": float(sum(totals.values())),
                "team_reward_mean_per_agent": float(np.mean(list(totals.values()))),
                "per_agent_reward": totals,
                "red_session_hosts_by_agent": host_counts,
                "invalid_action_rate": float(invalid_actions / max(total_actions, 1)),
                "action_counts": dict(action_counts),
            })
    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    summary = {
        "episodes": len(records),
        "mean_team_reward_sum": float(np.mean([r["team_reward_sum"] for r in records])),
        "std_team_reward_sum": float(np.std([r["team_reward_sum"] for r in records], ddof=1))
        if len(records) > 1 else 0.0,
        "mean_invalid_action_rate": float(np.mean([r["invalid_action_rate"] for r in records])),
        "mean_red_session_hosts": float(np.mean([
            sum(r["red_session_hosts_by_agent"].values()) for r in records
        ])),
        "episode_records": str(output),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
