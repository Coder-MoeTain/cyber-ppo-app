# Risk-Aware Hierarchical Multi-Agent Reinforcement Learning for Coordinated Autonomous Cyber Defense

Research package for CAGE Challenge 4 / CybORG enterprise cyber defence.
Includes a mock environment, CTDE MAPPO scaffold, CybORG adapter boundary,
statistical utilities and risk-conditioned hierarchical manager/worker policy.

## Quick start
Install development dependencies with pip install -e ".[dev]", then run
pytest -q. For a software smoke test run:
PYTHONPATH=src python scripts/train_mock_mappo.py --steps 256 --rollout-steps 64 --seed 42

The mock training command is not a CybORG benchmark and is not scientific
evidence for a paper.

## Status
- MAPPO trainer: implemented; mock-tested.
- Hierarchical manager/worker networks: implemented and unit-tested; not yet
  integrated into a PPO rollout/update loop.
- CybORG adapter: integration boundary implemented; live simulator validation pending.
- Dashboard: static frontend prototype with illustrative values only.

See docs/PUBLICATION_READINESS.md for remaining research and reproducibility gates.
