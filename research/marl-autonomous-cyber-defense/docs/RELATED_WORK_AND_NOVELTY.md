# Related work and novelty positioning

## CAGE Challenge 4 is an established benchmark, not a new contribution

CAGE Challenge 4 (CC4) is an enterprise-network multi-agent cyber-defence
environment with five Blue defenders, randomized network/host configurations,
mission phases, asymmetric local observations, and constrained inter-agent
communication. The official challenge documentation and source code must be
cited as the benchmark specification.

The challenge has already been studied in peer-reviewed work:

- Kiely et al., "Exploring the Efficacy of Multi-Agent Reinforcement Learning
  for Autonomous Cyber Defence: A CAGE Challenge 4 Perspective," AAAI 2025,
  pp. 28907-28913, DOI: 10.1609/aaai.v39i28.35158.
- Kiely et al., "CAGE challenge 4: A scalable multi-agent reinforcement
  learning gym for autonomous cyber defence," AI Magazine, 46(3), e70021,
  2025, DOI: 10.1002/aaai.70021.

The published challenge analysis reports that heuristic submissions
outperformed the MARL submissions considered. It also identifies dynamic
network-configuration inference and invalid actions as weaknesses for MARL
agents. The reported top-four comparison includes a heuristic score of
-113 ± 35 and a MARL score of -193 ± 84 on the standard CC4 evaluation.
These results are a direct warning against presenting "MAPPO applied to
CC4" as sufficient novelty or evidence of effectiveness.

## Required novelty claim discipline

Do not claim that this project is the first MARL cyber-defence framework,
the first use of MAPPO in CC4, or that hierarchy necessarily improves
performance. Those claims are unsupported.

A defensible candidate contribution to test is:

1. A risk-conditioned hierarchical manager/worker policy under decentralized
   execution and centralized training.
2. Explicit handling of randomized topology and invalid actions through
   observation-aware action masking and topology-generalization evaluation.
3. A measured security/availability/intervention-cost trade-off under
   risk-conditioned defensive prioritization.
4. Robustness analysis across adversary and environment shifts, not only
   average reward on the default scenario.

These are hypotheses, not established contributions, until experiments
show statistically and practically meaningful gains over strong baselines.

## Research questions to preregister

- RQ1: Does risk-conditioned hierarchical coordination improve team utility
  and security/availability outcomes over a strong rule-based heuristic,
  Independent PPO, and MAPPO on the standard CC4 distribution?
- RQ2: Does the method generalize better to held-out network sizes and host
  configurations than the baselines?
- RQ3: Does risk conditioning improve response prioritization without causing
  excessive containment, service disruption, or false-positive interventions?

## Minimum baseline and evaluation discipline

A strong CC4 heuristic comparison is mandatory. The repository's current
alert-guided Analyse/Sleep policy is only a preliminary baseline; strengthen
and validate it against the published challenge analysis. Do not describe it
as an exact reproduction of a competition submission unless original source
and settings are available. Include Independent
PPO, MAPPO, and the proposed RA-HMAPPO under matched training interactions,
scenario distributions, action masks, and evaluation budgets.

Evaluate standard CC4 plus predeclared shifts based on the challenge literature:
constant network size, improved decoy detection, increased phishing,
stealthy Red, and aggressive Red. Keep test scenario seeds/configurations
disjoint from training and model selection. Report per-agent and team rewards,
service availability, critical-zone impact, detection/mitigation behavior,
invalid-action rate, unnecessary interventions, and communication cost.

## Bibliography

```bibtex
@inproceedings{kiely2025exploring,
  title={Exploring the Efficacy of Multi-Agent Reinforcement Learning for Autonomous Cyber Defence: A CAGE Challenge 4 Perspective},
  author={Kiely, Mitchell and Ahiskali, Metin and Borde, Etienne and Bowman, Benjamin and Bowman, David and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39}, number={28}, pages={28907--28913}, year={2025},
  doi={10.1609/aaai.v39i28.35158}
}

@article{kiely2025cage,
  title={CAGE challenge 4: A scalable multi-agent reinforcement learning gym for autonomous cyber defence},
  author={Kiely, Mitchell and Ahiskali, Metin and Borde, Etienne and Bowman, Benjamin and others},
  journal={AI Magazine}, volume={46}, number={3}, pages={e70021},
  year={2025}, doi={10.1002/aaai.70021}
}
```
