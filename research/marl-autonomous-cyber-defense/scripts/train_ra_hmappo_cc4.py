"""Train RA-HMAPPO on the pinned CAGE Challenge 4 environment."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import torch
import time

from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from autocd.algorithms.hierarchical_trainer import RAHMAPPOConfig, RiskAwareHierarchicalTrainer
from autocd.envs.cyborg_adapter import CybORGAdapter
from autocd.envs.cc4_tasks import CC4TaskActionMapper
from autocd.risk_model import RiskPredictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--risk-checkpoint")
    p.add_argument("--risk-mode", choices=["predicted", "constant"], default="predicted")
    p.add_argument("--task-mapping", choices=["semantic", "unconstrained"], default="semantic")
    p.add_argument("--reward-aggregation", choices=["sum", "mean"], default="sum")
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--episode-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.steps < 1 or args.rollout_steps < 1 or args.episode_steps < 1:
        raise ValueError("step counts must be positive")

    if args.risk_mode == "predicted" and not args.risk_checkpoint:
        p.error("--risk-checkpoint is required when --risk-mode=predicted")
    if args.risk_mode == "constant" and args.risk_checkpoint:
        p.error("--risk-checkpoint must be omitted when --risk-mode=constant")

    env = CybORGAdapter(
        scenario_generator=EnterpriseScenarioGenerator(steps=args.episode_steps),
        seed=args.seed,
    )
    try:
        risk_fn = (RiskPredictor.load(args.risk_checkpoint, device=args.device)
                   if args.risk_mode == "predicted" else lambda agent, obs, info: 0.5)
        config = RAHMAPPOConfig(num_tasks=len(CC4TaskActionMapper.TASKS))
        task_mapper = CC4TaskActionMapper(env.env) if args.task_mapping == "semantic" else None
        trainer = RiskAwareHierarchicalTrainer(
            env, risk_fn, config, seed=args.seed, device=args.device,
            reward_aggregation=args.reward_aggregation, task_action_mapper=task_mapper,
        )
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize(args.device)
        training_start = time.perf_counter()
        history = trainer.train(args.steps, args.rollout_steps)
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize(args.device)
        training_wall_seconds = time.perf_counter() - training_start
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": trainer.model.state_dict(),
            "agents": list(trainer.agents),
            "observation_dim": trainer.obs_dim,
            "action_dim": next(iter(trainer.action_dims.values())),
            "global_state_dim": trainer.global_state_dim,
            "config": asdict(config),
            "seed": args.seed,
            "risk_checkpoint": str(args.risk_checkpoint) if args.risk_checkpoint else None,
            "risk_mode": args.risk_mode,
            "task_mapping": args.task_mapping,
            "reward_aggregation": args.reward_aggregation,
            "environment": "CAGE-Challenge-4",
            "cyborg_version": "4.0",
            "training_steps": args.steps,
            "training_wall_seconds": training_wall_seconds,
            "training_steps_per_second": args.steps / max(training_wall_seconds, 1e-9),
        }, output)
        output.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps({
            "checkpoint": str(output),
            "history": str(output.with_suffix(".history.json")),
            "environment": "CAGE-Challenge-4",
            "training_steps": args.steps,
            "seed": args.seed,
        }, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
