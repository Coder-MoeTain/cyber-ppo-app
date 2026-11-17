# CC4 training and evaluation entry points

Use disjoint seed sets for risk-model training and validation, then keep final
policy-evaluation seeds untouched until model selection is frozen.

```bash
python scripts/collect_cc4_risk_dataset.py --seeds 11 22 33 44 55 66 77 88 99 --steps 500 --output data/risk_train.npz
python scripts/collect_cc4_risk_dataset.py --seeds 201 202 203 204 --steps 500 --output data/risk_validation.npz
python scripts/train_risk_model.py --train-data data/risk_train.npz --validation-data data/risk_validation.npz --output artifacts/cc4_risk.pt

python scripts/train_ra_hmappo_cc4.py --risk-checkpoint artifacts/cc4_risk.pt --steps 1000000 --rollout-steps 256 --seed 42 --output artifacts/ra_hmappo_seed42.pt
python scripts/train_ippo_cc4.py --steps 1000000 --rollout-steps 256 --seed 42 --output artifacts/ippo_seed42.pt

python scripts/evaluate_ra_hmappo_cc4.py --checkpoint artifacts/ra_hmappo_seed42.pt --risk-checkpoint artifacts/cc4_risk.pt --seeds 501 502 503 504 505 506 507 508 509 510 --output results/ra_hmappo_seed42.jsonl
python scripts/evaluate_baseline_cc4.py --baseline ippo --checkpoint artifacts/ippo_seed42.pt --seeds 501 502 503 504 505 506 507 508 509 510 --output results/ippo_seed42.jsonl
python scripts/evaluate_baseline_cc4.py --baseline heuristic --seeds 501 502 503 504 505 506 507 508 509 510 --output results/heuristic.jsonl
```

The default episode horizon is 500 steps, matching the challenge-style
evaluation horizon. Run all methods with the same held-out scenario seeds,
episode horizon and interaction budget. Repeat training over the preregistered
training seed set; the examples above show only one seed for illustration.

The RA-HMAPPO evaluator uses deterministic manager and worker decisions with
task-specific action masks. It writes episode-level JSONL and reports summed
and per-agent rewards, task/action counts, invalid-action rate and final
red-session host counts by defended zone. It does not claim to measure service
availability or mitigation latency; add and validate those metrics before
using them as paper outcomes.

The heuristic is an original simple alert-guided baseline, not an exact
reproduction of a published competition agent. Report that limitation.
