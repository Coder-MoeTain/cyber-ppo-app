# Research Specification

## Provisional title
Risk-Aware Hierarchical Multi-Agent Reinforcement Learning for Autonomous Cyber Defense in Dynamic Enterprise Networks.

## Aim
Design and evaluate a risk-aware hierarchical MARL framework for coordinated autonomous defense in simulated enterprise networks.

## Research questions
- RQ1: Can specialized defensive agents coordinate autonomous decisions?
- RQ2: Does learned risk-aware hierarchy outperform standard MAPPO?
- RQ3: How does coordination affect critical-asset compromise and service availability?
- RQ4: How does the method generalize to held-out attack scenarios?
- RQ5: What compute and communication overhead does coordination introduce?

## Hypotheses
- H1: Proposed coordination improves defense effectiveness over Independent PPO.
- H2: Risk-aware coordination reduces critical-asset compromise versus standard MAPPO.
- H3: The method improves or preserves availability at comparable security effectiveness.
- H4: Hierarchy helps under simultaneous attack events.
- H5: Benefits are achieved with measurable, bounded overhead.

These are hypotheses, not results.

## Baselines
Rule-based Blue policy, PPO, Independent PPO, standard MAPPO, proposed RA-HMAPPO.

## Outcomes
Attack success rate, critical asset compromise, service availability, time to mitigation, unnecessary-action rate, and compute/communication overhead.
