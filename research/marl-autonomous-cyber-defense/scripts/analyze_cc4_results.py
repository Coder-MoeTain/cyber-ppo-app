"""Paired, two-level bootstrap analysis for the standard CC4 matrix."""
import argparse
import glob
import json
from pathlib import Path
import numpy as np

from autocd.statistics import paired_permutation_test, holm_bonferroni


def load_records(pattern, require_training_seed=True):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No evaluation files matched: {pattern}")
    records = []
    for filename in files:
        with open(filename, encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if "seed" not in row or "team_reward_sum" not in row:
                        raise ValueError(f"Malformed episode record in {filename}")
                    if require_training_seed and row.get("training_seed") is None:
                        raise ValueError(f"Missing training_seed in {filename}")
                    records.append(row)
    return records


def model_matrix(records):
    training_seeds = sorted({int(r["training_seed"]) for r in records})
    evaluation_seeds = sorted({int(r["seed"]) for r in records})
    matrix = np.full((len(training_seeds), len(evaluation_seeds)), np.nan)
    train_index = {s: i for i, s in enumerate(training_seeds)}
    eval_index = {s: i for i, s in enumerate(evaluation_seeds)}
    for row in records:
        i, j = train_index[int(row["training_seed"])], eval_index[int(row["seed"])]
        if np.isfinite(matrix[i, j]):
            raise ValueError("Duplicate training-seed/evaluation-seed episode")
        matrix[i, j] = float(row["team_reward_sum"])
    if not np.isfinite(matrix).all():
        raise ValueError("Model evaluation matrix is incomplete")
    return training_seeds, evaluation_seeds, matrix


def heuristic_vector(records):
    seeds = sorted({int(r["seed"]) for r in records})
    index = {s: i for i, s in enumerate(seeds)}
    values = np.full(len(seeds), np.nan)
    for row in records:
        j = index[int(row["seed"])]
        if np.isfinite(values[j]):
            raise ValueError("Duplicate heuristic evaluation seed")
        values[j] = float(row["team_reward_sum"])
    if not np.isfinite(values).all():
        raise ValueError("Heuristic evaluation vector is incomplete")
    return seeds, values


def validate_expected_matrix(training_seeds, evaluation_seeds, *, expected_training, expected_evaluation):
    """Reject partial matrices even when the observed rows form a rectangular grid."""
    actual_training = list(map(int, training_seeds))
    actual_evaluation = list(map(int, evaluation_seeds))
    expected_training = list(map(int, expected_training))
    expected_evaluation = list(map(int, expected_evaluation))
    if len(set(expected_training)) != len(expected_training):
        raise ValueError("Expected training seeds contain duplicates")
    if len(set(expected_evaluation)) != len(expected_evaluation):
        raise ValueError("Expected evaluation seeds contain duplicates")
    if actual_training != sorted(expected_training):
        raise ValueError(
            f"Training seed set mismatch: expected {sorted(expected_training)}, got {actual_training}"
        )
    if actual_evaluation != sorted(expected_evaluation):
        raise ValueError(
            f"Evaluation seed set mismatch: expected {sorted(expected_evaluation)}, got {actual_evaluation}"
        )


def two_level_bootstrap_difference(a, b, *, n_boot=10000, seed=1234):
    """Resample independent training seeds and paired evaluation seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 2 or min(a.shape) < 2:
        raise ValueError("paired matrices must have equal 2-D shapes with >=2 rows/columns")
    rng = np.random.default_rng(seed)
    n_train, n_eval = a.shape
    samples = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        ti = rng.integers(0, n_train, n_train)
        ei = rng.integers(0, n_eval, n_eval)
        samples[k] = a[np.ix_(ti, ei)].mean() - b[np.ix_(ti, ei)].mean()
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_difference": float(a.mean() - b.mean()),
        "ci_95": [float(lo), float(hi)],
        "n_training_seeds": int(n_train),
        "n_evaluation_seeds": int(n_eval),
        "bootstrap_replicates": int(n_boot),
    }


def model_vs_heuristic_bootstrap(model, heuristic, *, n_boot=10000, seed=1234):
    model, heuristic = np.asarray(model, float), np.asarray(heuristic, float)
    if model.ndim != 2 or heuristic.ndim != 1 or model.shape[1] != heuristic.size:
        raise ValueError("model matrix and heuristic vector must share evaluation-seed dimension")
    if min(model.shape) < 2:
        raise ValueError("at least two training and evaluation seeds are required")
    rng = np.random.default_rng(seed)
    n_train, n_eval = model.shape
    samples = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        ti = rng.integers(0, n_train, n_train)
        ei = rng.integers(0, n_eval, n_eval)
        samples[k] = model[np.ix_(ti, ei)].mean() - heuristic[ei].mean()
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_difference": float(model.mean() - heuristic.mean()),
        "ci_95": [float(lo), float(hi)],
        "n_training_seeds": int(n_train),
        "n_evaluation_seeds": int(n_eval),
        "bootstrap_replicates": int(n_boot),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ra-glob", required=True)
    p.add_argument("--ippo-glob", required=True)
    p.add_argument("--mappo-glob", required=True)
    p.add_argument("--heuristic-file", required=True)
    p.add_argument("--alert-only-file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--expected-training-seeds", type=int, nargs="+",
                   default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    p.add_argument("--expected-evaluation-seeds", type=int, nargs="+",
                   default=list(range(501, 601)))
    args = p.parse_args()
    if args.bootstrap < 100:
        raise ValueError("bootstrap must be at least 100")

    ra_records = load_records(args.ra_glob)
    ippo_records = load_records(args.ippo_glob)
    mappo_records = load_records(args.mappo_glob)
    heuristic_records = load_records(args.heuristic_file, require_training_seed=False)
    alert_only_records = load_records(args.alert_only_file, require_training_seed=False)
    ra_train, eval_seeds, ra = model_matrix(ra_records)
    ippo_train, ippo_eval, ippo = model_matrix(ippo_records)
    mappo_train, mappo_eval, mappo = model_matrix(mappo_records)
    h_seeds, h = heuristic_vector(heuristic_records)
    alert_seeds, alert_only = heuristic_vector(alert_only_records)
    validate_expected_matrix(
        ra_train, eval_seeds,
        expected_training=args.expected_training_seeds,
        expected_evaluation=args.expected_evaluation_seeds,
    )
    if ra_train != ippo_train or ra_train != mappo_train:
        raise ValueError("RA-HMAPPO, IPPO and MAPPO must use the same training seeds")
    if (eval_seeds != ippo_eval or eval_seeds != mappo_eval or eval_seeds != h_seeds
            or eval_seeds != alert_seeds):
        raise ValueError("All methods must use exactly the same held-out evaluation seeds")

    comparisons = {}
    p_values = []
    labels = []
    for name, left, right in (
        ("RA-HMAPPO_vs_IPPO", ra, ippo),
        ("RA-HMAPPO_vs_MAPPO", ra, mappo),
    ):
        left_seed_means, right_seed_means = left.mean(axis=1), right.mean(axis=1)
        test = paired_permutation_test(left_seed_means, right_seed_means, seed=args.seed)
        ci = two_level_bootstrap_difference(
            left, right, n_boot=args.bootstrap, seed=args.seed
        )
        diff = left_seed_means - right_seed_means
        effect = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
        comparisons[name] = {**ci, "paired_seed_permutation": test, "cohens_dz": effect}
        p_values.append(test["p_value"])
        labels.append(name)

    # Average across independent training runs for each paired evaluation seed.
    ra_eval_mean = ra.mean(axis=0)
    heuristic_diff = ra_eval_mean - h
    heuristic_test = paired_permutation_test(ra_eval_mean, h, seed=args.seed)
    heuristic_ci = model_vs_heuristic_bootstrap(
        ra, h, n_boot=args.bootstrap, seed=args.seed
    )
    comparisons["RA-HMAPPO_vs_heuristic"] = {
        **heuristic_ci,
        "paired_evaluation_seed_permutation": heuristic_test,
        "cohens_dz": float(
            heuristic_diff.mean() / (heuristic_diff.std(ddof=1) + 1e-12)
        ),
    }
    p_values.append(heuristic_test["p_value"])
    labels.append("RA-HMAPPO_vs_heuristic")

    alert_only_test = paired_permutation_test(ra_eval_mean, alert_only, seed=args.seed)
    alert_only_ci = model_vs_heuristic_bootstrap(
        ra, alert_only, n_boot=args.bootstrap, seed=args.seed
    )
    alert_only_diff = ra_eval_mean - alert_only
    comparisons["RA-HMAPPO_vs_alert_only"] = {
        **alert_only_ci,
        "paired_evaluation_seed_permutation": alert_only_test,
        "cohens_dz": float(
            alert_only_diff.mean() / (alert_only_diff.std(ddof=1) + 1e-12)
        ),
    }
    p_values.append(alert_only_test["p_value"])
    labels.append("RA-HMAPPO_vs_alert_only")

    adjusted = holm_bonferroni(p_values)
    for name, p_adj in zip(labels, adjusted):
        comparisons[name]["holm_adjusted_p_value"] = p_adj

    def mean_if_present(values):
        values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
        return float(np.mean(values)) if values else None

    def summary(records):
        return {
            "episodes": len(records),
            "mean_team_reward_sum": float(np.mean([r["team_reward_sum"] for r in records])),
            "mean_invalid_action_rate": mean_if_present(
                [r.get("invalid_action_rate") for r in records]
            ),
            "mean_final_red_session_hosts": mean_if_present([
                sum(r.get("red_session_hosts_by_agent", {}).values())
                for r in records if r.get("red_session_hosts_by_agent") is not None
            ]),
            "mean_red_session_host_timesteps_per_step": mean_if_present([
                r.get("red_session_host_timesteps_per_step") for r in records
            ]),
            "mean_restore_actions_per_episode": mean_if_present([
                r.get("operational_action_counts", {}).get("restore_actions")
                for r in records
            ]),
            "mean_traffic_control_actions_per_episode": mean_if_present([
                r.get("operational_action_counts", {}).get("traffic_control_actions")
                for r in records
            ]),
            "mean_policy_inference_ms": mean_if_present([
                r.get("inference_latency_ms", {}).get("mean")
                for r in records if r.get("inference_latency_ms")
            ]),
            "mean_policy_inference_p95_ms": mean_if_present([
                r.get("inference_latency_ms", {}).get("p95")
                for r in records if r.get("inference_latency_ms")
            ]),
        }

    result = {
        "primary_endpoint": "episode cumulative sum of five Blue-agent rewards",
        "training_seeds": ra_train,
        "evaluation_seeds": eval_seeds,
        "methods": {
            "RA-HMAPPO": summary(ra_records),
            "IPPO": summary(ippo_records),
            "MAPPO": summary(mappo_records),
            "heuristic": summary(heuristic_records),
            "alert_only": summary(alert_only_records),
        },
        "planned_comparisons": comparisons,
        "caveats": [
            "Statistical inference is only as valid as the independence of training and evaluation seeds.",
            "Heuristic comparison averages RA-HMAPPO across training seeds before paired evaluation-seed permutation.",
            "These scripts do not correct for scenario-variant multiplicity; preregister additional tests before running variants.",
            "Reward and red-session host counts are not substitutes for separately validated service-availability and mitigation-latency metrics.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
