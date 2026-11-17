# Draft Experimental Protocol

Compare rule-based, PPO, Independent PPO, MAPPO, and proposed RA-HMAPPO.

Control scenario distributions, observation permissions, training interactions, evaluation episodes, hardware, and checkpoint selection. Record deviations.

Scenario groups: routine operation, single event, concurrent events, critical-asset exposure, lateral movement, service disruption, held-out configurations, and constrained communication. Use only supported scenarios/actions from the pinned CybORG version.

Use multiple independent seeds (target >=10 where feasible). Report mean, standard deviation, confidence intervals, effect sizes, and justified paired tests. Preserve episode-level metrics.

Ablate learned coordinator, risk features, risk-aware reward, task allocation, coordination reward, and communication. Keep evaluation scenarios separate from training and validation.
