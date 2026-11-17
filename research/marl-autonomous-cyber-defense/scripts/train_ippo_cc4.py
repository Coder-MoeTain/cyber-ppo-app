"""Train Independent PPO on pinned CAGE Challenge 4."""
import argparse
from dataclasses import asdict
from pathlib import Path
import torch

from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from autocd.algorithms.independent_ppo import IPPOConfig, IndependentPPOTrainer
from autocd.envs.cyborg_adapter import CybORGAdapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--episode-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    env = CybORGAdapter(
        scenario_generator=EnterpriseScenarioGenerator(steps=args.episode_steps),
        seed=args.seed,
    )
    try:
        cfg = IPPOConfig()
        trainer = IndependentPPOTrainer(env, cfg, seed=args.seed, device=args.device)
        history = trainer.train(args.steps, args.rollout_steps)
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
        }, output)
        torch.save(history, output.with_suffix(".history.pt"))
        print({"checkpoint": str(output), "steps": args.steps, "seed": args.seed})
    finally:
        env.close()


if __name__ == "__main__":
    main()
