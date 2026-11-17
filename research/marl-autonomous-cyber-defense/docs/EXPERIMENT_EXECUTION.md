# Full experiment execution guide

This guide separates workflows that can be tested in ordinary CI from benchmark workloads that require reserved compute.

## Required compute and runtime

The standard matrix defaults to 10 independent training seeds, 1,000,000 environment steps per seed for each of RA-HMAPPO, IPPO, and MAPPO, and 100 held-out evaluation seeds. This is a substantial workload (30 million learned-policy training interactions plus evaluation and risk-data collection). Do not run it on a short-lived unit-test runner. Reserve a persistent machine with the pinned CC4 dependencies and sufficient disk for checkpoints and episode records.

## Reproducible invocation

From this directory, after installing the pinned CC4 validation dependencies and the research package:

```bash
DEVICE=cuda STEPS=1000000 ROLLOUT_STEPS=256 EPISODE_STEPS=500 \
OUT=artifacts/standard_cc4 RESULTS=results/standard_cc4 \
bash scripts/run_standard_cc4_matrix.sh
```

Use `DEVICE=cpu` only for smoke tests or when no compatible accelerator is available. Record the machine model, accelerator, driver, CUDA/PyTorch versions, CPU/RAM, OS, wall-clock duration, and peak memory.

## Preflight

1. Run the full unit suite and compile checks.
2. Run the pinned live CC4 adapter and trainer smoke tests.
3. Run the official upstream CC4 tests and resolve failures.
4. Inspect risk-label prevalence and confirm both classes exist in train and validation splits.
5. Verify train, validation, and final evaluation seed sets are disjoint.
6. Review baseline behavior and document the Team-UC-inspired policy's fidelity limitations.
7. Freeze hypotheses, metrics, exclusion rules, ablations, and statistical tests.

## Outputs and acceptance

The matrix writes risk-model data/checkpoint under `data/` and `artifacts/`, trained checkpoints per seed, episode-level JSONL records, summaries, and paired analysis JSON under `results/`. Preserve the complete output directory and its SHA-256 manifest.

A successful process exit is not enough. Inspect every expected file, verify each training seed and each of the 100 evaluation seeds is present exactly once per model checkpoint, validate finite metrics, and compare logged environment-step counts with the requested budget. Reject incomplete matrices rather than silently analyzing partial results.

## Held-out evaluation

The current standard matrix uses the same 100 episode seeds (501–600) for every method. Do not use these seeds for hyperparameter selection, early stopping, or checkpoint selection. Any scenario-shift evaluation requires its own predeclared configs and held-out seed split.

## Ablations and manuscript

The current scripts implement the main standard-CC4 matrix; they do not yet implement every ablation listed in `EXPERIMENT_PROTOCOL.md`. Do not label the planned ablations complete until each variant has an explicit configuration, runner support, matched-budget runs, raw outputs, and analysis. The manuscript draft in `MANUSCRIPT_DRAFT.md` intentionally leaves the Results and Conclusion pending actual evidence.
