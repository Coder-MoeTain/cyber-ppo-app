# Full experiment execution guide

This guide separates ordinary CI validation from long-running benchmark work. It also specifies disjoint data roles for the supervised risk estimator.

## Required compute and runtime

The standard matrix defaults to 10 independent training seeds, 1,000,000 environment steps per seed for each of RA-HMAPPO, IPPO, and MAPPO, and 100 held-out evaluation seeds. This is a substantial workload (30 million learned-policy training interactions plus risk-data collection and evaluation). Do not run it on a short-lived unit-test runner. Reserve a persistent machine with the pinned CC4 dependencies and sufficient disk for checkpoints and episode records.

## Reproducible invocation

From this directory, after installing the pinned CC4 validation dependencies and the research package:

```bash
DEVICE=cuda STEPS=1000000 ROLLOUT_STEPS=256 EPISODE_STEPS=500 \
OUT=artifacts/standard_cc4 RESULTS=results/standard_cc4 \
bash scripts/run_standard_cc4_matrix.sh
```

Use `DEVICE=cpu` only for smoke tests or when no compatible accelerator is available. Record the machine model, accelerator, driver, CUDA/PyTorch versions, CPU/RAM, OS, wall-clock duration, and peak memory.

## Risk-estimator data separation

The matrix now collects four disjoint seed partitions:

- Training: seeds 11–99 (offline supervised model fitting).
- Model selection: seeds 201–204 (early stopping only).
- Calibration: seeds 301–320 (scalar temperature fitting only).
- Final risk-model test: seeds 401–420 (reported AUROC, AUPRC, Brier, ECE and prevalence only).

The RL policy evaluation set remains seeds 501–600. The risk-model test set and policy evaluation set are distinct. The risk estimator's final test metrics must not be used for hyperparameter selection or calibration. The script rejects overlapping split seed IDs and missing classes. Because time steps within one episode are correlated, report episode-level provenance and do not interpret the sample count as independent episode count.

## Preflight

1. Run the full unit suite and compile checks.
2. Run the pinned live CC4 adapter and trainer smoke tests.
3. Run the official upstream CC4 tests and resolve failures.
4. Inspect risk-label prevalence and confirm both classes exist in every required split.
5. Verify train, validation, calibration, risk-test and policy-evaluation seed sets are disjoint.
6. Review baseline behavior and document the Team-UC-inspired policy's fidelity limitations.
7. Freeze hypotheses, metrics, exclusion rules, ablations, and statistical tests.

## Outputs and acceptance

The matrix writes risk-model data/checkpoint and a separate risk metrics JSON under `data/` and `artifacts/`, trained checkpoints per seed, episode-level JSONL records, summaries, and paired analysis JSON under `results/`. Preserve the complete output directory and its SHA-256 manifest.

The analysis script now requires exactly the preregistered 10 training seeds and 100 policy evaluation seeds by default. It rejects a rectangular but partial matrix, duplicate seed pairs, missing rows, or mismatched method seed sets. A successful process exit is not enough: inspect expected artifacts, finite metrics, and logged environment-step counts.

## Held-out evaluation

The standard policy matrix uses the same 100 episode seeds (501–600) for every method. Do not use these seeds for hyperparameter selection, early stopping, or checkpoint selection. Any scenario-shift evaluation requires its own versioned configs and held-out seed split.

## Ablations and manuscript

The current scripts implement the main standard-CC4 matrix; they do not yet implement every ablation listed in `EXPERIMENT_PROTOCOL.md`. Do not label planned ablations complete until each variant has explicit configuration, runner support, matched-budget runs, raw outputs, and analysis. The manuscript draft intentionally leaves Results and Conclusion pending actual evidence.
