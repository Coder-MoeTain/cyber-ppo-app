"""Train Independent PPO on pinned CAGE Challenge 4."""
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
import time
from training_telemetry import TrainingTelemetry

from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from autocd.algorithms.independent_ppo import IPPOConfig, IndependentPPOTrainer
from autocd.envs.cyborg_adapter import CybORGAdapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--episode-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    p.add_argument("--progress-file")
    p.add_argument("--rolling-checkpoint")
    args = p.parse_args()
    env = CybORGAdapter(
        scenario_generator=EnterpriseScenarioGenerator(steps=args.episode_steps),
        seed=args.seed,
    )
    try:
        cfg = IPPOConfig()
        trainer = IndependentPPOTrainer(env, cfg, seed=args.seed, device=args.device)
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize(args.device)
        telemetry = TrainingTelemetry(
            args.progress_file,
            args.rolling_checkpoint or str(args.output) + ".latest.pt",
            trainer.model, trainer.optimizer, args.steps, args.device,
            metadata={
                "algorithm": "IPPO",
                "seed": args.seed, "episode_steps": args.episode_steps,
                "rollout_steps": args.rollout_steps, "device": args.device,
                "agents": list(trainer.agents),
                "observation_dim": getattr(trainer, "obs_dim", None),
                "action_dims": getattr(trainer, "action_dims", None),
                "global_state_dim": getattr(trainer, "global_state_dim", None),
                "config": asdict(config) if "config" in locals() else (asdict(cfg) if "cfg" in locals() else {}),
            },
        )
        training_start = time.perf_counter()
        history = trainer.train(args.steps, args.rollout_steps, on_update=telemetry.on_update)
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize(args.device)
        training_wall_seconds = time.perf_counter() - training_start
        telemetry.finish("completed", {"training_wall_seconds": training_wall_seconds})
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": trainer.model.state_dict(),
            "agents": list(trainer.agents),
            "observation_dims": trainer.observation_dims,
            "action_dims": trainer.action_dims,
            "hidden_dim": cfg.hidden_dim,
            "config": asdict(cfg),
            "seed": args.seed,
            "environment": "CAGE-Challenge-4",
            "cyborg_version": "4.0",
            "training_steps": args.steps,
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "training_wall_seconds": training_wall_seconds,
            "training_steps_per_second": args.steps / max(training_wall_seconds, 1e-9),
        }, output)
        torch.save(history, output.with_suffix(".history.pt"))
        print({"checkpoint": str(output), "steps": args.steps, "seed": args.seed})
    finally:
        env.close()


if __name__ == "__main__":
    main()
