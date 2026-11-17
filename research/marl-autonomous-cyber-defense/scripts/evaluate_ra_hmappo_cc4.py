"""Evaluate an RA-HMAPPO checkpoint on held-out CC4 seeds."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from autocd.algorithms.hierarchical import RiskAwareHierarchicalActorCritic
from autocd.envs.cyborg_adapter import CybORGAdapter
from autocd.envs.cc4_tasks import CC4TaskActionMapper
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
    p.add_argument("--risk-checkpoint")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--episode-steps", type=int, default=500)
    p.add_argument("--device", default="cpu")
    p.add_argument("--profile-inference", action="store_true",
                   help="Synchronize accelerators and record per-agent policy inference latency.")
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
        risk_mode = checkpoint.get("risk_mode", "predicted")
        if risk_mode == "predicted":
            risk_path = args.risk_checkpoint or checkpoint.get("risk_checkpoint")
            if not risk_path:
                raise RuntimeError("Predicted-risk checkpoint is required by this policy")
            risk_fn = RiskPredictor.load(risk_path, device=args.device)
        elif risk_mode == "constant":
            if args.risk_checkpoint:
                raise RuntimeError("Do not supply a risk checkpoint for constant-risk ablation")
            risk_fn = lambda agent, obs, info: 0.5
        else:
            raise RuntimeError(f"Unknown checkpoint risk_mode: {risk_mode}")
        task_mapping = checkpoint.get("task_mapping", "semantic")
        task_mapper = CC4TaskActionMapper(env.env) if task_mapping == "semantic" else None

        for seed in args.seeds:
            observations, infos = env.reset(seed=seed)
            totals = {a: 0.0 for a in env.possible_agents}
            action_counts = Counter()
            task_counts = Counter()
            invalid_actions = 0
            total_actions = 0
            steps = 0
            inference_ms = []
            red_host_timesteps = {a: 0 for a in env.possible_agents}
            for _ in range(args.episode_steps):
                masks = env.action_masks()
                actions = {}
                for agent in env.agents:
                    obs = np.asarray(observations[agent], dtype=np.float32)
                    risk = risk_fn(agent, obs, infos.get(agent, {}))
                    mask = masks[agent]
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=args.device)
                    risk_tensor = torch.tensor([risk], dtype=torch.float32, device=args.device)
                    task_action_masks = (
                        task_mapper(agent, obs, infos.get(agent, {}), mask)
                        if task_mapper is not None else np.repeat(mask[None, :], model.num_tasks, axis=0)
                    )
                    task_mask = torch.as_tensor(
                        task_action_masks.any(axis=1), dtype=torch.bool, device=args.device
                    )
                    if args.profile_inference and str(args.device).startswith("cuda"):
                        torch.cuda.synchronize(args.device)
                    inference_start = time.perf_counter()
                    with torch.no_grad():
                        manager_dist = model.manager_distribution(
                            agent, obs_tensor, risk_tensor, task_mask
                        )
                        task_tensor = manager_dist.probs.argmax(-1)
                        task = int(task_tensor.item())
                        worker_mask = torch.as_tensor(
                            task_action_masks[task], dtype=torch.bool, device=args.device
                        )
                        worker_dist = model.worker_distribution(
                            agent, obs_tensor, risk_tensor, task_tensor, worker_mask
                        )
                        action = int(worker_dist.probs.argmax(-1).item())
                    if args.profile_inference:
                        if str(args.device).startswith("cuda"):
                            torch.cuda.synchronize(args.device)
                        inference_ms.append((time.perf_counter() - inference_start) * 1000.0)
                    actions[agent] = action
                    task_counts[CC4TaskActionMapper.TASKS[task]] += 1
                    total_actions += 1
                    invalid_actions += int(not bool(mask[action]))
                    label = env.env.action_labels(agent)[action]
                    category = str(label).split()[0].replace("[Invalid]", "Invalid")
                    action_counts[category] += 1

                observations, rewards, terms, truncs, infos = env.step(actions)
                for agent, reward in rewards.items():
                    totals[agent] += float(reward)
                steps += 1
                for a in env.possible_agents:
                    red_host_timesteps[a] += red_host_count(env, env.env.hosts(a))
                if all(bool(terms.get(a, False) or truncs.get(a, False))
                       for a in env.possible_agents):
                    break

            host_counts = {
                a: red_host_count(env, env.env.hosts(a)) for a in env.possible_agents
            }
            records.append({
                "seed": int(seed),
                "training_seed": int(checkpoint["seed"]),
                "risk_mode": checkpoint.get("risk_mode", "predicted"),
                "task_mapping": checkpoint.get("task_mapping", "semantic"),
                "reward_aggregation": checkpoint.get("reward_aggregation", "sum"),
                "steps": steps,
                "team_reward_sum": float(sum(totals.values())),
                "team_reward_mean_per_agent": float(np.mean(list(totals.values()))),
                "per_agent_reward": totals,
                "red_session_hosts_by_agent": host_counts,
                "red_session_host_timesteps_by_agent": red_host_timesteps,
                "red_session_host_timesteps_total": int(sum(red_host_timesteps.values())),
                "red_session_host_timesteps_per_step": float(
                    sum(red_host_timesteps.values()) / max(steps, 1)
                ),
                "operational_action_counts": {
                    "restore_actions": int(action_counts.get("Restore", 0)),
                    "traffic_control_actions": int(
                        action_counts.get("BlockTrafficZone", 0) + action_counts.get("AllowTrafficZone", 0)
                    ),
                    "remove_actions": int(action_counts.get("Remove", 0)),
                    "decoy_deployments": int(action_counts.get("DeployDecoy", 0)),
                },
                "inference_latency_ms": ({
                    "mean": float(np.mean(inference_ms)),
                    "median": float(np.median(inference_ms)),
                    "p95": float(np.quantile(inference_ms, 0.95)),
                    "p99": float(np.quantile(inference_ms, 0.99)),
                    "n_agent_decisions": int(len(inference_ms)),
                } if inference_ms else None),
                "invalid_action_rate": float(invalid_actions / max(total_actions, 1)),
                "action_counts": dict(action_counts),
                "task_counts": dict(task_counts),
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
