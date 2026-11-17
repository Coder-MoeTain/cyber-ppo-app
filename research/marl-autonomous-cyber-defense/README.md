# Risk-Aware Hierarchical Multi-Agent Reinforcement Learning for Coordinated Autonomous Cyber Defense

Research implementation for the CAGE Challenge 4 (CC4) enterprise cyber-defence
benchmark. The package includes CTDE MAPPO, Independent PPO, a risk-conditioned
hierarchical policy and PPO trainer, an offline-supervised risk estimator
pipeline, a Team UC-inspired heuristic plus a simple alert-only comparator,
statistical utilities and a professional static dashboard prototype.

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest -q
PYTHONPATH=src python scripts/train_mock_mappo.py --steps 256 --rollout-steps 64 --seed 42
```

These mock tests validate software interfaces only. They are not CC4
benchmark results.

## Pinned CC4 integration

The official simulator source is pinned to the exact commit in
`configs/cc4.lock.yaml`. The live integration workflow checks the official
upstream CC4 tests, adapter reset, action masks, same-seed replay and a real
simulator step. See `requirements-cc4-validation.txt`.

## Research workflows

- `scripts/collect_cc4_risk_dataset.py`: collect local observations and
  offline red-session labels on designated training seeds.
- `scripts/train_risk_model.py`: train and calibrate the local risk estimator
  on disjoint training/validation datasets.
- `scripts/train_ra_hmappo_cc4.py`: train RA-HMAPPO on the pinned simulator.
- `scripts/train_ippo_cc4.py`: train the Independent PPO baseline.
- `scripts/train_mappo_cc4.py`: train the CTDE MAPPO baseline.
- `scripts/run_standard_cc4_matrix.sh`: reproducible multi-seed standard-CC4 experiment matrix.
- `scripts/analyze_cc4_results.py`: paired seed-level comparisons, bootstrap intervals and Holm correction.
- `scripts/evaluate_ra_hmappo_cc4.py` and
  `scripts/evaluate_baseline_cc4.py`: evaluate on held-out seeds and write
  episode-level JSONL records.

The Team UC-inspired heuristic is an independent reimplementation, not an exact
reproduction of a published CC4 competition agent; it omits malicious-file flags.
The simpler alert-only comparator is also available. The dashboard is a static frontend with
illustrative values and is not connected to live experiment telemetry.

See `docs/PUBLICATION_READINESS.md` before making any publication claims.
