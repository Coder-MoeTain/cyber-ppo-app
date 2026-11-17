"""End-to-end RA-HMAPPO software smoke test on real CC4 (not a benchmark run)."""
import argparse
import json
import math

from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from autocd.algorithms.hierarchical_trainer import (
    RAHMAPPOConfig, RiskAwareHierarchicalTrainer,
)
from autocd.envs.cc4_tasks import CC4TaskActionMapper
from autocd.envs.cyborg_adapter import CybORGAdapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    env = CybORGAdapter(
        scenario_generator=EnterpriseScenarioGenerator(steps=500),
        seed=args.seed,
    )
    try:
        config = RAHMAPPOConfig(
            update_epochs=1, minibatch_size=max(2, args.steps // 2),
            hidden_dim=32, num_tasks=len(CC4TaskActionMapper.TASKS),
        )
        trainer = RiskAwareHierarchicalTrainer(
            env, lambda agent, observation, info: 0.5, config,
            seed=args.seed, task_action_mapper=CC4TaskActionMapper(env.env),
        )
        rollout = trainer.collect_rollout(args.steps)
        metrics = trainer.update(rollout)
        values = [metrics[k] for k in (
            "actor_loss", "critic_loss", "manager_entropy", "worker_entropy"
        )]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("Non-finite RA-HMAPPO metrics")
        print(json.dumps({
            "environment": "real_CybORG_CC4",
            "purpose": "integration_smoke_only_not_benchmark_evidence",
            "steps": args.steps,
            "agents": list(trainer.agents),
            "observation_dim": trainer.obs_dim,
            "action_dim": next(iter(trainer.action_dims.values())),
            "num_tasks": config.num_tasks,
            "metrics": metrics,
        }, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
