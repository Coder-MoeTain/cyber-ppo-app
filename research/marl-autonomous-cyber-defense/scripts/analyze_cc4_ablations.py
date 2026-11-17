"""Paired multi-seed analysis for the RA-HMAPPO ablation matrix."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from autocd.statistics import paired_permutation_test, holm_bonferroni
from analyze_cc4_results import (
    load_records, model_matrix, two_level_bootstrap_difference,
    validate_expected_matrix,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-glob", required=True)
    parser.add_argument("--no-risk-glob", required=True)
    parser.add_argument("--no-mapping-glob", required=True)
    parser.add_argument("--mean-reward-glob", required=True)
    parser.add_argument("--risk-rule-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--expected-training-seeds", type=int, nargs="+",
                        default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    parser.add_argument("--expected-evaluation-seeds", type=int, nargs="+",
                        default=list(range(501, 601)))
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("bootstrap must be at least 100")

    patterns = {
        "full": args.full_glob,
        "no_risk": args.no_risk_glob,
        "no_semantic_mapping": args.no_mapping_glob,
        "mean_team_reward": args.mean_reward_glob,
        "risk_rule_manager": args.risk_rule_glob,
    }
    matrices, records_by_name = {}, {}
    training_seeds = evaluation_seeds = None
    for name, pattern in patterns.items():
        records = load_records(pattern)
        train, evaluation, matrix = model_matrix(records)
        validate_expected_matrix(
            train, evaluation,
            expected_training=args.expected_training_seeds,
            expected_evaluation=args.expected_evaluation_seeds,
        )
        if training_seeds is None:
            training_seeds, evaluation_seeds = train, evaluation
        elif train != training_seeds or evaluation != evaluation_seeds:
            raise ValueError(f"{name} does not use the exact full-model seed matrix")
        matrices[name] = matrix
        records_by_name[name] = records

    comparisons, p_values, labels = {}, [], []
    for name in ("no_risk", "no_semantic_mapping", "mean_team_reward", "risk_rule_manager"):
        left, right = matrices["full"], matrices[name]
        full_seed_means, ablation_seed_means = left.mean(axis=1), right.mean(axis=1)
        test = paired_permutation_test(full_seed_means, ablation_seed_means, seed=args.seed)
        ci = two_level_bootstrap_difference(left, right, n_boot=args.bootstrap, seed=args.seed)
        paired_diff = full_seed_means - ablation_seed_means
        effect = float(paired_diff.mean() / (paired_diff.std(ddof=1) + 1e-12))
        comparisons[f"full_vs_{name}"] = {
            **ci, "paired_training_seed_permutation": test, "cohens_dz": effect,
        }
        labels.append(f"full_vs_{name}")
        p_values.append(test["p_value"])

    adjusted = holm_bonferroni(p_values)
    for label, p_adj in zip(labels, adjusted):
        comparisons[label]["holm_adjusted_p_value"] = p_adj

    summaries = {}
    for name, records in records_by_name.items():
        values = np.asarray([r["team_reward_sum"] for r in records], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite episode reward in {name}")
        def mean_metric(items):
            valid = [float(x) for x in items if x is not None and np.isfinite(float(x))]
            return float(np.mean(valid)) if valid else None

        summaries[name] = {
            "episodes": int(values.size),
            "mean_team_reward_sum": float(values.mean()),
            "std_team_reward_sum": float(values.std(ddof=1)),
            "mean_invalid_action_rate": mean_metric(
                [r.get("invalid_action_rate") for r in records]
            ),
            "mean_red_session_host_timesteps_per_step": mean_metric([
                r.get("red_session_host_timesteps_per_step") for r in records
            ]),
            "mean_restore_actions_per_episode": mean_metric([
                r.get("operational_action_counts", {}).get("restore_actions")
                for r in records
            ]),
            "mean_traffic_control_actions_per_episode": mean_metric([
                r.get("operational_action_counts", {}).get("traffic_control_actions")
                for r in records
            ]),
            "mean_policy_inference_ms": mean_metric([
                r.get("inference_latency_ms", {}).get("mean")
                for r in records if r.get("inference_latency_ms")
            ]),
        }

    result = {
        "primary_endpoint": "episode cumulative sum of five Blue-agent rewards",
        "training_seeds": training_seeds,
        "evaluation_seeds": evaluation_seeds,
        "methods": summaries,
        "planned_comparisons": comparisons,
        "caveats": [
            "Ablations use paired training seeds and common held-out evaluation seeds.",
            "Confidence intervals resample training and evaluation seeds at two levels.",
            "No superiority claim is warranted without reviewing raw episodes and assumptions.",
            "Unconstrained task mapping removes semantic task-specific worker masks; it does not remove the manager/worker hierarchy.",
            "The flat MAPPO baseline in the main matrix is the non-hierarchical comparator.",
            "The deterministic risk-rule manager is a fixed monitor/investigate threshold policy; it is not a learned manager.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
