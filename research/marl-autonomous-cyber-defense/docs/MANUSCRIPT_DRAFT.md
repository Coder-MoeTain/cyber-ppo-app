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

1. A risk-conditioned manager/worker policy for coordinated multi-agent cyber defense.
2. A semantic task layer mapping high-level defensive intents to legal CC4 actions through official action masks.
3. A reproducible evaluation and ablation protocol with held-out seeds and paired inference.

These are implementation contributions; empirical novelty and efficacy remain to be established by the results.

## 2. Related Work

This section must be completed with a systematic, citation-verified review before submission. It should cover: (i) RL and MARL for cyber defense; (ii) PPO, MAPPO and centralized training with decentralized execution; (iii) hierarchical RL and options; (iv) risk-sensitive and risk-aware decision-making; and (v) CAGE/CC4 benchmark results and heuristic baselines.

The final version must explicitly discuss published CC4 findings that heuristic agents can outperform submitted MARL agents, and must distinguish this implementation's Team-UC-inspired baseline from the original competition implementation. Do not claim exact reproduction: the current wrapper does not expose all malicious-file features described in the published heuristic.

## 3. Problem Formulation

Model the environment as a partially observable multi-agent stochastic game. Let agents be indexed by (i \in \{1,\ldots,N\}), local observations by (o_t^i), legal action sets by (A_t^i\), and local reward streams by (r_t^i). The centralized training state is used only by critic-side components where permitted by the simulator interface; decentralized actors receive local observations and the declared risk estimate.

The team return aggregation must be frozen before final evaluation. The current experiment runner uses the sum of the five Blue-agent rewards as its primary team endpoint. All secondary outcomes and privileged simulator-state measurements must be documented with their actor-observation boundary.

## 4. Proposed Method

### 4.1 Risk estimator

A supervised binary estimator predicts whether a red session is present in the local zone. Ground-truth simulator state is used offline to create labels; the trained estimator receives local Blue observations and an agent identity encoding at inference. The final manuscript must report label prevalence, train/validation split provenance, AUROC, AUPRC, Brier score, expected calibration error, calibration method, and failure cases.

### 4.2 Hierarchical policy

The manager selects one of six semantic task families: monitor, investigate, remediate, recover, isolate, or deceive. A worker selects a concrete action from the chosen task's legal action mask. The implementation uses separate manager and worker policy distributions and PPO objectives.

### 4.3 Risk-aware objective

The risk signal conditions the manager and worker policy networks. The final paper must state the exact risk normalization, model checkpoint provenance, and whether uncertainty is represented. No claim of formal risk guarantees should be made unless separately proven.

### 4.4 Coordination and communication

The current experiments use the wrapper's default empty inter-agent messages. The study therefore does **not** claim learned communication, message passing, or communication-efficiency improvements.

## 5. Experimental Design

The pinned CC4 commit, software environment, training seeds, held-out evaluation seeds, episode horizon, hardware, and run configurations must be archived. The planned primary comparison includes RA-HMAPPO, IPPO, MAPPO, and the Team-UC-inspired heuristic. An alert-only heuristic may be reported as a secondary comparator.

The target matrix is ten independent training seeds and 100 held-out evaluation episodes per trained checkpoint and scenario, with identical held-out episode seeds across methods. Scenario-shift experiments must use versioned configuration files and a split that is not used for hyperparameter selection.

### Metrics

- Primary: cumulative sum of the five Blue-agent reward streams per episode.
- Security: red-session host counts and any additional CC4-supported compromise/mission outcomes with clearly stated measurement semantics.
- Operational: service availability, green-user disruption, invalid-action rate, unnecessary containment/restoration, and mitigation latency where instrumentation permits.
- Efficiency: inference latency, training wall-clock time, peak memory, and checkpoint size.
- Risk estimator: AUROC, AUPRC, Brier score, ECE, calibration curve, and positive-label prevalence.

Unavailable metrics must be marked as unavailable rather than inferred from reward.

## 6. Ablation Studies

The final evaluation should include at minimum:
1. RA-HMAPPO with the trained risk estimator versus a constant-risk control.
2. Hierarchical manager/worker policy versus a flat policy baseline.
3. Learned manager task selection versus deterministic task allocation.
4. Sum versus mean team reward aggregation.
5. Alternative semantic task mappings and action-mask diagnostics.
6. Risk estimator calibration/uncertainty variants.

Each ablation must have a precise implementation definition, matched interaction budget, independent training seeds, identical held-out episodes, and a predeclared primary comparison. Results must not be described as an ablation if only a different checkpoint or training budget was used.

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
