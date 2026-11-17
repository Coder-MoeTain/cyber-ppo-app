"""Run the published H-MARL CAGE-4 checkpoint on paired episode seeds.

This must run in an isolated environment compatible with the upstream project
(CybORG plus Ray/RLlib 2.10). It does not modify the upstream checkout.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
from statistics import stdev


UPSTREAM_COMMIT = "6fe960f5931f2d7dc5ef522f09ca4a179f377ee4"
UPSTREAM_REPO = "https://github.com/adityavs14/Hierarchical-MARL"
EPISODE_HORIZON = 500


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream H-MARL module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_variant(hmarl_root, variant):
    variant_root = hmarl_root / "h-marl-3policy"
    run_dir = variant_root / ("master" if variant == "meta" else "subpolicies")
    evaluation_path = run_dir / "evaluation.py"
    submission_path = run_dir / "submission.py"
    if not evaluation_path.is_file() or not submission_path.is_file():
        raise FileNotFoundError(f"Incomplete upstream H-MARL {variant} checkout: {run_dir}")

    sys.path.insert(0, str(variant_root))
    sys.path.insert(0, str(run_dir))
    evaluation = load_module(f"hmarl_{variant}_evaluation", evaluation_path)
    submission_module = load_module(f"hmarl_{variant}_submission", submission_path)
    return variant_root, run_dir, evaluation, submission_module.Submission


def run_one(evaluation, submission, variant_root, seed, output_dir):
    original_cwd = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    try:
        # Upstream agent checkpoints are resolved relative to h-marl-3policy/.
        os.chdir(variant_root)
        evaluation.stdev = lambda values: stdev(values) if len(values) > 1 else 0.0
        with contextlib.redirect_stdout(buffer):
            evaluation.run_evaluation(
                submission, str(output_dir) + os.sep,
                max_eps=1, write_to_file=False, seed=int(seed),
            )
    finally:
        os.chdir(original_cwd)
    (output_dir / "run.log").write_text(buffer.getvalue(), encoding="utf-8")
    match = re.search(
        r"Average reward is:\s*([-+0-9.eE]+)\s+with a standard deviation",
        buffer.getvalue(),
    )
    if not match:
        raise RuntimeError(
            f"Could not parse H-MARL reward for seed {seed}; captured output tail:\n"
            + buffer.getvalue()[-3000:]
        )
    mean_per_agent = float(match.group(1))
    return {
        "seed": int(seed),
        "training_seed": None,
        "method": f"H-MARL-{variant}",
        # Upstream evaluation sums the per-step mean across Blue agents.
        # Multiply by five to match this project's sum of five Blue reward streams.
        "team_reward_sum": mean_per_agent * 5.0,
        "team_reward_mean_per_agent": mean_per_agent,
        "episode_horizon": EPISODE_HORIZON,
        "reward_semantics": "upstream mean Blue reward per step, summed over episode, multiplied by five",
        "upstream_repository": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmarl-root", required=True)
    parser.add_argument("--variant", choices=["meta", "expert"], default="meta")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    hmarl_root = Path(args.hmarl_root).resolve()
    if not (hmarl_root / ".git").exists():
        raise ValueError("--hmarl-root must be a checked-out upstream H-MARL Git repository")
    import subprocess
    actual_commit = subprocess.check_output(
        ["git", "-C", str(hmarl_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != UPSTREAM_COMMIT:
        raise ValueError(
            f"Expected upstream H-MARL commit {UPSTREAM_COMMIT}, got {actual_commit}"
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Evaluation seeds must be unique")

    variant_root, _, evaluation, submission_class = load_variant(hmarl_root, args.variant)
    submission = submission_class()
    records = []
    for seed in args.seeds:
        records.append(run_one(
            evaluation, submission, variant_root, seed,
            Path(args.output).resolve().parent / f"hmarl_{args.variant}_seed{seed}",
        ))

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8"
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "upstream_commit": UPSTREAM_COMMIT,
        "episode_count": len(records),
        "evaluation_seeds": list(map(int, args.seeds)),
        "reward_conversion": "team_reward_sum = 5 * upstream reward.mean",
        "caveat": "Published H-MARL uses a custom feature-rich wrapper and communication; information inputs differ from this project's BlueFlatWrapper policy interface.",
        "records": str(output),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
