# v0.3.0 — MAPPO and CybORG integration milestone

This release milestone adds the next research implementation layer for Risk-Aware Hierarchical Multi-Agent Reinforcement Learning for Coordinated Autonomous Cyber Defense.

## Implemented in the local v0.3.0 package

- Cooperative CTDE MAPPO trainer with decentralized per-agent actors and a centralized critic.
- Action masking and rollout collection for multi-agent policies.
- GAE handling that distinguishes true termination from truncation and prevents traces crossing reset boundaries.
- CybORG/CAGE Challenge 4 adapter using EnterpriseScenarioGenerator and BlueFlatWrapper.
- CC4 validation script for agent count, observation/action dimensions and action masks.
- Paired bootstrap confidence intervals, paired sign-flip permutation tests, and Holm-Bonferroni correction.
- Mock MAPPO smoke-training script and expanded tests.
- Statistical analysis and CC4 integration documentation.
- CI workflow for tests and mock PPO/MAPPO smoke checks.

## Local validation recorded for this package

- pytest: 19 passed.
- Mock MAPPO training smoke test completed with finite actor/critic losses and entropy.
- Mock PPO training smoke test completed.
- Python compileall passed.

## Important limitations

The live CybORG/CAGE Challenge 4 simulator has not yet been installed or executed in this environment. The adapter contract is tested with a fake backend; real simulator compatibility remains an explicit validation gate. The package is therefore a research implementation milestone, not a publication-ready validated result.

The full v0.3.0 source package was generated locally as MARL-Autonomous-Cyber-Defense-v0.3.0.zip. This repository commit records the release scope and validation status; it does not claim that the local ZIP has been uploaded as a repository asset.
