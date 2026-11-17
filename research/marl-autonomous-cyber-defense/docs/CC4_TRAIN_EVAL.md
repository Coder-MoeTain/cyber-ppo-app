# CC4 training and evaluation entry points

After pinning the official simulator and training the offline risk model:

```bash
python scripts/train_ra_hmappo_cc4.py --risk-checkpoint artifacts/cc4_risk.pt --steps 1000000 --rollout-steps 256 --seed 42 --output artifacts/ra_hmappo_seed42.pt
python scripts/evaluate_ra_hmappo_cc4.py --checkpoint artifacts/ra_hmappo_seed42.pt --risk-checkpoint artifacts/cc4_risk.pt --seeds 501 502 503 504 505 --output results/ra_hmappo_seed42.jsonl
```

The evaluation script uses deterministic actions and writes episode-level
JSONL. It reports summed and per-agent rewards, invalid-action rate,
action-category counts and final red-session host counts by defended zone.
It does not claim to measure service availability or mitigation latency;
those require separately validated instrumentation.

Run the same evaluation harness for every baseline with identical test seeds.
Do not compare algorithms using different episode seeds or interaction budgets.
