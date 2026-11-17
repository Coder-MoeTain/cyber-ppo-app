# Risk-Aware Hierarchical Multi-Agent Reinforcement Learning for Coordinated Autonomous Cyber Defense in Enterprise Networks

**Manuscript status: evidence-pending research draft.** This document is a structured manuscript draft, not a submission-ready paper. No benchmark outcomes, confidence intervals, or superiority claims may be inserted until the preregistered CC4 experiments have run and the raw episode records have been independently checked.

## Abstract

Coordinated cyber defense in enterprise networks requires multiple defenders to allocate limited actions across monitoring, investigation, containment, recovery, and deception. Independent policies can produce inconsistent responses, while centralized policies may not respect operational constraints at each defender. This study investigates a risk-aware hierarchical multi-agent reinforcement learning approach (RA-HMAPPO) that separates high-level defensive task selection from low-level action selection and conditions both levels on an estimate of local compromise risk.

The proposed method is implemented against the pinned CAGE Challenge 4 (CC4) simulator and is designed for comparison with Independent PPO, centralized-training/decentralized-execution MAPPO, and a stateful Team-UC-inspired heuristic. The evaluation protocol specifies independent training seeds, a held-out evaluation set, paired statistical analysis, calibration assessment, and ablation studies. **Quantitative results are intentionally withheld pending execution of the complete experimental matrix.** The paper will report security outcomes, operational availability, invalid actions, intervention costs, inference overhead, and generalization under predeclared scenario shifts.

**Keywords:** multi-agent reinforcement learning; cyber defense; hierarchical reinforcement learning; risk-aware control; MAPPO; autonomous response; CAGE Challenge 4.

## 1. Introduction

Enterprise cyber defense is a sequential decision problem involving partial observability, multiple network zones, constrained defensive actions, and competing objectives. A response that reduces attacker activity may also disrupt legitimate services. Multi-agent reinforcement learning (MARL) offers a framework for coordinating distributed defenders, but learned policies must be evaluated against strong operational heuristics and under reproducible scenario variation.

This work studies whether an explicit hierarchy and a calibrated local risk signal improve coordinated defensive decision-making. The central hypothesis is that a manager policy can select a defensive task appropriate to the current risk context while a worker policy selects an available simulator action consistent with that task.

### Research questions

- RQ1: Does RA-HMAPPO improve team utility and security/availability outcomes over IPPO, MAPPO, and a strong stateful heuristic under matched budgets?
- RQ2: Does risk conditioning improve defensive prioritization without increasing unnecessary intervention or service disruption?
- RQ3: Which parts of the hierarchy and risk-estimation pipeline account for any observed performance differences?
- RQ4: How robust is the approach to predeclared CC4 scenario shifts?

### Contributions to be tested

1. A locally conditioned risk signal learned from privileged offline labels but restricted to local observations at inference, with separate model-selection, calibration and final-test episode splits.
2. A controlled risk-conditioned hierarchical PPO implementation with semantic task-specific action masks and explicit risk-off/task-mapping/reward-aggregation ablations.
3. A reproducible CAGE-4 evaluation comparing the proposed policy with IPPO, MAPPO, a documented stateful heuristic, and—if reproducible within the project budget—the prior H-MARL approach.

These are hypotheses and engineering contributions, not established novelty claims. The project must demonstrate a measurable benefit from the risk signal beyond prior hierarchical CAGE-4 methods. If it cannot, the manuscript should be reframed as a replication/negative-result study rather than claiming a new algorithm.

## 2. Related Work

Hierarchical MARL is not itself a novel contribution in this setting. Singh et al. [1] proposed H-MARL Expert and H-MARL Meta for CAGE-4, using a master policy to select PPO sub-policies for tasks such as investigation and recovery. Their work also showed the value of richer security indicators and reported clean-host, recovery-precision and false-positive metrics. The present implementation differs in using one learned manager/worker actor-critic conditioned on a scalar local-zone risk estimate trained from offline simulator labels; it does not train independently transferable sub-policies. The empirical question must therefore be whether calibrated local risk conditioning adds value beyond a risk-blind hierarchy, not whether hierarchy alone is new.

The CAGE-4 challenge evaluation [2] is a necessary benchmark reference: published competition results showed that carefully engineered heuristics could outperform submitted MARL agents. This makes the Team-UC-inspired comparator important, but it must not be described as an exact reproduction because the current wrapper does not expose all malicious-file features described in the published policy. The evaluation should also include the earlier H-MARL method [1] or a carefully documented reproduction if feasible; IPPO and MAPPO alone are insufficient hierarchical baselines.

More recent work further narrows the novelty claim. Posterior-conditioned MARL for adaptive cyber defense [3] introduces a shared attack-context belief inferred online from multi-agent trajectories and evaluates adaptation to mixed and switching attacks in CAGE-4. That differs from this implementation's offline-supervised, per-agent local risk score: the current code has no shared belief, online attack-regime inference, or learned inter-agent communication. Safety-contract graph MARL [4] also argues that reward-only policies can violate operational downtime, false-positive and firewall-change budgets. The present implementation does not yet impose those explicit operational constraints; it must not claim operational safety or deployability from reward and red-session counts alone.

MAPPO provides a strong cooperative-policy baseline [5], while the options framework provides the temporal-abstraction background for hierarchical task selection [6]. The related-work section and final claims must be revised again after the full literature search and after the latest 2026 work is checked for publication status and code availability.

## 3. Problem Formulation

Model the environment as a partially observable multi-agent stochastic game. Let agents be indexed by (i \in \{1,\ldots,N\}), local observations by (o_t^i), legal action sets by (A_t^i\), and local reward streams by (r_t^i). The centralized training state is used only by critic-side components where permitted by the simulator interface; decentralized actors receive local observations and the declared risk estimate.

The team return aggregation must be frozen before final evaluation. The current experiment runner uses the sum of the five Blue-agent rewards as its primary team endpoint. All secondary outcomes and privileged simulator-state measurements must be documented with their actor-observation boundary.

## 4. Proposed Method

### 4.1 Risk estimator

A supervised binary estimator predicts whether a red session is present in the local zone. Ground-truth simulator state is used offline to create labels; the trained estimator receives local Blue observations and an agent identity encoding at inference. The implemented protocol separates training seeds (121–140), model-selection seeds (201–220), temperature-calibration seeds (301–320), and final risk-model test seeds (401–420). The final policy evaluation uses a separate seed range (501–600). The training script reports test AUROC, AUPRC, Brier score and ECE, together with 95% episode-cluster bootstrap intervals. These seed assignments and the final sample/episode counts must be preserved in the experiment manifest.

### 4.2 Hierarchical policy

The manager selects one of six semantic task families: monitor, investigate, remediate, recover, isolate, or deceive. A worker selects a concrete action from the chosen task's legal action mask. The implementation uses separate manager and worker policy distributions and PPO objectives.

### 4.3 Risk-aware objective

The risk signal conditions the manager and worker policy networks. The final paper must state the exact risk normalization, model checkpoint provenance, and whether uncertainty is represented. No claim of formal risk guarantees should be made unless separately proven.

### 4.4 Coordination and communication

The current experiments use the wrapper's default empty inter-agent messages. The study therefore does **not** claim learned communication, message passing, or communication-efficiency improvements.

## 5. Experimental Design

The pinned CC4 commit, software environment, training seeds, held-out evaluation seeds, episode horizon, hardware, and run configurations must be archived. The planned primary comparison includes RA-HMAPPO, IPPO, MAPPO, and the Team-UC-inspired heuristic; the alert-only heuristic is included as a secondary comparator. The main analysis script pairs both heuristic comparators on the same 100 held-out seeds.

The target matrix is ten independent training seeds and 100 held-out evaluation episodes per trained checkpoint and scenario, with identical held-out episode seeds across methods. Scenario-shift experiments must use versioned configuration files and a split that is not used for hyperparameter selection.

### Metrics

- Primary: cumulative sum of the five Blue-agent reward streams per episode.
- Security: red-session host counts and any additional CC4-supported compromise/mission outcomes with clearly stated measurement semantics.
- Security/operational proxies currently implemented: red-session host-time exposure, final red-session host counts, invalid-action rate, and Restore/Remove/traffic-control/decoy action counts.
- Efficiency currently implemented: per-agent policy inference latency (mean/median/p95/p99) and training wall-clock time/throughput.
- Not yet instrumented: direct Green-user service availability, downtime duration, false-positive recovery cost, peak memory, and action-to-mitigation latency. These must be measured directly or explicitly listed as unavailable; do not infer them from reward or action counts.
- Risk estimator: AUROC, AUPRC, Brier score, ECE, calibration curve, and positive-label prevalence.

Unavailable metrics must be marked as unavailable rather than inferred from reward.

## 6. Ablation Studies

The current executable ablation runner implements:
1. RA-HMAPPO with the trained risk estimator versus a constant-risk (0.5) control.
2. Learned manager versus a deterministic risk-threshold manager (investigate above threshold; monitor otherwise, with legal-task fallback).
3. Semantic task-specific action masks versus unconstrained task-to-action mapping (all legal base actions available to each task).
4. Sum versus mean team-reward aggregation.
5. Hierarchical manager/worker versus flat MAPPO in the main baseline matrix.

Deterministic task allocation and risk-calibration/uncertainty variants remain planned, not implemented. The prior H-MARL approach [1] is a required additional comparator if a faithful reproduction is feasible. Each ablation must use the same ten training seeds, the same 100 held-out evaluation seeds, and matched environment interaction budgets.

## 7. Statistical Analysis

Report per-method distributions, means and standard deviations, 95% confidence intervals, paired effect sizes, and multiplicity-adjusted p-values for predeclared comparisons. Resampling units must respect independent training runs and paired evaluation episodes; timesteps within an episode must not be treated as independent observations. Publish raw per-episode records and the analysis script.

The current analysis code uses paired seed-level permutation tests, a two-level bootstrap for learned-policy comparisons, and Holm-Bonferroni adjustment. The statistical plan should be independently reviewed before final submission, especially the learned-policy-versus-heuristic comparison and any scenario-variant multiplicity.

## 8. Results

**Pending execution. Do not populate this section with mock values or estimates.**

| Method | Training seeds | Held-out episodes | Mean team reward | 95% CI | Availability | Invalid action rate |
|---|---:|---:|---:|---:|---:|---:|
| RA-HMAPPO | Pending | Pending | Pending | Pending | Pending | Pending |
| IPPO | Pending | Pending | Pending | Pending | Pending | Pending |
| MAPPO | Pending | Pending | Pending | Pending | Pending | Pending |
| Team-UC-inspired heuristic | N/A | Pending | Pending | Pending | Pending | Pending |

Insert only values reproducible from archived raw results. Include all failed runs, exclusions, and deviations from the preregistered protocol.

## 9. Discussion

This section must interpret actual findings against the hypotheses, compare the learned policies with the strong heuristic baseline, discuss operational trade-offs, and identify failure modes. A result in cumulative reward alone is insufficient to claim safer or more effective cyber defense.

## 10. Threats to Validity and Limitations

- Simulator validity: CC4 is a benchmark abstraction and does not establish effectiveness on operational enterprise networks.
- Baseline fidelity: the Team-UC-inspired policy is not an exact competition-source reproduction and lacks some published feature inputs.
- Risk-label privilege: offline labels use simulator ground truth; leakage into actor inputs must be ruled out.
- Generalization: conclusions are limited to evaluated scenario configurations and seed distributions.
- Metric coverage: service availability and mitigation latency require validated instrumentation.
- Communication: no learned inter-agent communication is evaluated.
- Statistical power: ten training seeds and 100 evaluation episodes are targets, not proof of adequate power; justify the final sample size.

## 11. Conclusion

To be written after the full matrix, ablations, robustness checks, and statistical review. The conclusion must reflect the observed evidence and must not claim publication readiness based on software completion alone.

## Reproducibility Statement

Release the exact research branch commit, pinned CC4 source commit, dependency lock, all run configurations, training and evaluation seeds, checkpoints, raw JSONL episode records, risk datasets or documented data-generation procedure, analysis outputs, test reports, and hardware/software environment. Clearly identify any artifacts that cannot be redistributed.


## References

[1] A. V. Singh, E. Rathbun, E. Graham, L. Oakley, S. Boboila, A. Oprea, and P. Chin, “Hierarchical Multi-agent Reinforcement Learning for Cyber Network Defense,” arXiv:2410.17351, 2024. https://arxiv.org/abs/2410.17351

[2] M. Kiely et al., “Exploring the Efficacy of Multi-Agent Reinforcement Learning for Autonomous Cyber Defence: A CAGE Challenge 4 Perspective,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 39, no. 28, pp. 28907–28913, 2025. doi:10.1609/aaai.v39i28.35158.

[3] A. Kazeminajafabadi, A. Ravari, N. D. Bastian, T. Lan, and M. Imani, “Posterior-conditioned multi-agent reinforcement learning for adaptive cyber defense,” *Artificial Intelligence*, vol. 360, article 104606, 2026. doi:10.1016/j.artint.2026.104606.

[4] J. L. L. de Jesus Silva, “Safety-Contract Graph Multi-Agent Reinforcement Learning for Autonomous Network Security Response,” arXiv:2606.13832, 2026. https://arxiv.org/abs/2606.13832

[5] C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, and Y. Wu, “The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games,” *Advances in Neural Information Processing Systems*, vol. 35, 2022. https://arxiv.org/abs/2103.01955

[6] R. S. Sutton, D. Precup, and S. Singh, “Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning,” *Artificial Intelligence*, vol. 112, no. 1–2, pp. 181–211, 1999. doi:10.1016/S0004-3702(99)00052-1
