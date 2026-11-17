# Novelty and related-work gap analysis

**Status: publication-critical.** The proposed RA-HMAPPO implementation is not the first hierarchical MARL system for CAGE-4 or cyber defense. Novelty must be established through controlled evidence, not architecture naming.

## Closest prior work

| Work | What it already contributes | Consequence for this project |
|---|---|---|
| Singh et al. (2024), *Hierarchical Multi-agent Reinforcement Learning for Cyber Network Defense*, arXiv:2410.17351 | H-MARL Expert and H-MARL Meta on CybORG CAGE-4; master policy selects task-specific PPO sub-policies; reports clean-host ratio, recovery precision and false positives; explores richer security indicators and transfer. | Hierarchical manager/subtask decomposition is prior art. Reproduce it or explain source/config incompatibility. Do not claim first hierarchical cyber-defense MARL. |
| Kiely et al. (2025), *Exploring the Efficacy of Multi-Agent Reinforcement Learning for Autonomous Cyber Defence: A CAGE Challenge 4 Perspective*, AAAI 2025 | Official CC4 evaluation and comparisons that highlight the strength of engineered heuristic policies. | Include strong, transparent heuristic baselines and explain fidelity limits. Reward-only learned policies must not be assumed superior. |
| Kazeminajafabadi et al. (2026), *Posterior-conditioned multi-agent reinforcement learning for adaptive cyber defense*, Artificial Intelligence 360, 104606 | Shared attack-context belief inferred online from multi-agent trajectories; adapts to mixed and switching CAGE-4 attacks without test-time retraining. | A local scalar risk estimate is materially different, but the paper is a very close current comparator. Explain local/offline versus shared/online belief conditioning and test under regime shifts. |
| Silva (2026), *Safety-Contract Graph Multi-Agent Reinforcement Learning for Autonomous Network Security Response*, arXiv:2606.13832 | CAGE-4 evaluation with explicit MTTR/downtime, false-positive response and firewall-change budgets, plus tail-risk and action screening. | The current project only logs intervention-count proxies; it does not enforce operational budgets. Do not claim safety or deployability until direct cost metrics and constraints are implemented. |
| Yu et al. (2022), *The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games*, NeurIPS 2022 | Strong PPO/MAPPO methodology and ablation practices for cooperative MARL. | MAPPO and IPPO are necessary baselines, not sufficient evidence of novelty. |

## Defensible research question

The current implementation is best positioned around this testable question:

> Does a separately trained and calibrated per-agent local-zone risk estimate improve coordinated CAGE-4 defense decisions beyond an otherwise matched hierarchical PPO policy, and under what attack-regime and operational-cost conditions?

The answer must be supported by:
1. Risk-estimator performance on episode-disjoint final test data, with episode-clustered confidence intervals.
2. Matched-seed full-model versus constant-risk ablation.
3. Matched-seed learned-manager versus deterministic risk-rule manager ablation.
4. Comparisons with IPPO, MAPPO, the Team-UC-inspired heuristic, alert-only baseline, and preferably a faithful H-MARL reproduction.
5. Scenario-shift evaluation with predeclared attack/topology changes.
6. Direct operational measures (Green-user service availability, downtime, false-positive recoveries, firewall changes) rather than reward-only or action-count proxies.

## Claims to avoid

- “The first hierarchical MARL framework for cyber defense.”
- “The first CAGE-4 hierarchical policy.”
- “Guaranteed safe,” “operationally safe,” or “deployable” without explicit constraints and measured service costs.
- “Learned inter-agent coordination/communication”: this implementation uses the wrapper's default empty messages.
- “Superior to state of the art” before matched experiments and a direct H-MARL comparison.
- Treating the Team-UC-inspired baseline as an exact reproduction.

## References

1. A. V. Singh et al., “Hierarchical Multi-agent Reinforcement Learning for Cyber Network Defense,” arXiv:2410.17351, 2024. https://arxiv.org/abs/2410.17351
2. M. Kiely et al., “Exploring the Efficacy of Multi-Agent Reinforcement Learning for Autonomous Cyber Defence: A CAGE Challenge 4 Perspective,” AAAI 2025, pp. 28907–28913. https://doi.org/10.1609/aaai.v39i28.35158
3. A. Kazeminajafabadi et al., “Posterior-conditioned multi-agent reinforcement learning for adaptive cyber defense,” *Artificial Intelligence*, vol. 360, article 104606, 2026. https://doi.org/10.1016/j.artint.2026.104606
4. J. L. L. de Jesus Silva, “Safety-Contract Graph Multi-Agent Reinforcement Learning for Autonomous Network Security Response,” arXiv:2606.13832, 2026. https://arxiv.org/abs/2606.13832
5. C. Yu et al., “The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games,” NeurIPS 2022. https://arxiv.org/abs/2103.01955
