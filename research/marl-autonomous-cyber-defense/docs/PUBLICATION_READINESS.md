# Publication Readiness Gate

Status: **NOT PUBLICATION READY**. This document tracks evidence, not aspirations.

## Required gates
- [ ] Pin a specific CybORG/CAGE-4 commit and environment dependencies.
- [ ] Validate native Blue agent observations/actions and legal-action masks.
- [ ] Prevent privileged simulator-state leakage at inference.
- [ ] Implement and independently validate rule-based, PPO, IPPO and MAPPO baselines.
- [ ] Implement a learned risk-aware hierarchical MARL coordinator (not deterministic routing).
- [ ] Specify a formal constrained Dec-POMDP and mathematically define risk, rewards, hierarchy.
- [ ] Train all methods on matched environment-interaction budgets.
- [ ] Use disjoint training/validation/test scenario configurations.
- [ ] Run independent seeds and publish raw per-episode results.
- [ ] Report attack success, asset compromise, availability, mitigation time and overhead.
- [ ] Conduct hierarchy/risk/communication/reward ablations.
- [ ] Quantify confidence intervals, effect sizes and statistical tests.
- [ ] Benchmark generalization to held-out enterprise configurations.
- [ ] Archive environment version, seeds, hyperparameters, checkpoints and evaluation scripts.
- [ ] Prepare paper with literature review, limitations, ethics and artifact instructions.
- [ ] Conduct independent reproducibility audit before submission.

## Evidence policy
Mock-enterprise tests validate software interfaces only. Do not cite them as
CybORG performance results. No claimed improvement, statistical significance
or publication acceptance exists until measured and independently checked.

## Immediate next milestone
Replace the intentionally unimplemented CybORG adapter with a version-pinned
adapter validated against official CAGE-4 examples. Train PPO/IPPO/MAPPO on
the real environment before proposing learned hierarchical coordination.
