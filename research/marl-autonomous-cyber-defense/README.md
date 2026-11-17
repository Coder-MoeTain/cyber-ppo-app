# Risk-Aware Hierarchical Multi-Agent Cyber Defense

Research scaffold for **Risk-Aware Hierarchical Multi-Agent Reinforcement Learning for Autonomous Cyber Defense in Dynamic Enterprise Networks**.

## Scope
- Environment target: CybORG / CAGE enterprise scenario.
- Baselines: rule-based, PPO, Independent PPO, MAPPO.
- Proposed method: risk-aware hierarchical coordination (RA-HMAPPO; provisional).
- Training paradigm: centralized training, decentralized execution (CTDE).

## Current status
This branch contains the Phase 1 foundation, a mock enterprise environment for software smoke tests, risk scoring, a deterministic task coordinator baseline, PyTorch actor/centralized-critic networks, PPO loss/GAE primitives, configuration, tests, and research protocols.

**Important:** the mock environment is not CybORG and cannot support claims about realistic cyber-defense performance. The CybORG adapter intentionally fails closed until its mappings are validated against a pinned CybORG commit. MAPPO training orchestration and RA-HMAPPO's learned hierarchy remain future implementation milestones.

## Quick start
Python 3.10–3.12:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/phase1_smoke_test.py
pytest
```

Official CybORG: https://github.com/cage-challenge/CybORG

Record the exact CybORG commit, Python version, dependencies, scenario configuration, and seeds before experiments.

## Research milestones
1. Validate official CybORG enterprise scenario and document observations/actions.
2. Implement version-specific observation/action adapters and multi-agent rollouts.
3. Train PPO and Independent PPO baselines.
4. Train standard MAPPO with centralized critic.
5. Implement learned high-level task allocation and risk-conditioned low-level policies.
6. Run matched comparisons, ablations, generalization, and statistical analysis.

## Safety
Use only CybORG, synthetic data, or isolated cyber ranges with explicit authorization. No real-world attack execution or production deployment is included.
