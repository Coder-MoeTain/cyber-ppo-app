# Risk-Aware Hierarchical Multi-Agent Cyber Defense

Research implementation for **Risk-Aware Hierarchical Multi-Agent Reinforcement Learning for Coordinated Autonomous Cyber Defense in Enterprise Networks**.

## v0.3.0 status
- CTDE MAPPO trainer with per-agent actors, centralized critic, dynamic action masking and GAE boundary handling.
- CybORG / CAGE Challenge 4 adapter boundary and validation script.
- MAPPO mock smoke script, statistical utilities, tests and publication-readiness gates.
- Responsive Sentinel MARL research dashboard preview in `dashboard/`.

**Validation boundary:** unit tests and smoke training use the mock environment. The live CybORG/CAGE Challenge 4 simulator has not yet been validated. Do not treat mock returns as benchmark results. Pin the exact CybORG commit and run `python scripts/validate_cyborg.py` before simulator experiments.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python scripts/train_mock_mappo.py --steps 8 --rollout-steps 4 --seed 42
```

## Dashboard preview
```bash
cd dashboard
python -m http.server 8080
```
Open http://localhost:8080. The UI uses illustrative values; the mock-run button does not start model training and live API integration is pending.

## Publication gates
See [docs/PUBLICATION_READINESS.md](docs/PUBLICATION_READINESS.md). Learned risk-aware hierarchy, Independent PPO baseline, live CybORG experiments, multi-seed comparisons, ablations and paper evidence remain outstanding.

Official simulator: https://github.com/cage-challenge/CybORG
