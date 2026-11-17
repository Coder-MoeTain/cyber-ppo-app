# Publication readiness gates

**Current classification: NOT publication ready.** Software implementation is
advancing, but no full CC4 training/evaluation results or paper evidence have
been produced in this release.

## Software and method

- [x] Five-agent CC4 adapter boundary with action-mask support.
- [x] CTDE MAPPO trainer and statistical utilities; mock-tested.
- [x] CC4 training/evaluation runners for MAPPO, IPPO and RA-HMAPPO plus a standard multi-seed matrix script.
- [x] Paired multi-seed analysis report generator with two-level bootstrap intervals and Holm correction.
- [x] Risk-conditioned hierarchical manager/worker policy architecture.
- [x] RA-HMAPPO clipped-PPO trainer with separate manager/worker log-probabilities;
  mock training tests added.
- [x] Six semantic CC4 manager options with task-specific worker action masks;
  unit tests and live smoke checks added.
- [x] Independent PPO baseline with decentralized critics; mock tests added.
- [x] Offline-supervised risk-estimation pipeline with held-out calibration
  metrics; data collection uses simulator ground truth only for labels.
- [x] Team UC-inspired stateful, action-mask-aware heuristic and simple
  alert-only comparator; neither is an exact competition-source reproduction.
- [ ] Validate the Team UC-inspired implementation against the published
  policy details and report its missing malicious-file features.
- [x] Official CAGE Challenge 4 source revision pinned in `configs/cc4.lock.yaml`.
- [x] Real CC4 adapter reset, same-seed replay, action-mask and one-step smoke validation.
- [x] Real CC4 offline risk-dataset collection smoke and RA-HMAPPO rollout/update smoke.
- [ ] Pass the official upstream CC4 test suite.
- [ ] Audit privileged risk-label semantics and held-out risk-model calibration.
- [ ] Empirically validate task ontology and ablate alternative option mappings.
- [ ] Validate reward aggregation and episode-boundary semantics on real CC4.
- [ ] Complete matched-budget training and evaluation for heuristic, IPPO,
  MAPPO and RA-HMAPPO.

## Experimental and manuscript

- [ ] Freeze hypotheses, primary endpoints and disjoint scenario seed/config splits.
- [ ] Train risk model on training seeds and report held-out calibration,
  AUROC/AUPRC, Brier score and ECE.
- [ ] Run at least 10 independent training seeds where feasible.
- [ ] Evaluate at least 100 episodes per method and scenario where feasible,
  including standard CC4 and predeclared generalization shifts.
- [ ] Report team/per-agent reward, red-session outcomes, availability,
  invalid-action rate, unnecessary intervention, latency and overhead.
- [ ] Run risk-input, hierarchy, task-mapping and reward-aggregation ablations.
- [ ] Archive raw episode records, configs, checkpoints, source provenance,
  statistical scripts and environment logs.
- [ ] Complete current novelty review, manuscript, verified citations,
  artifact license and clean-machine reproduction.

Do not present mock-environment returns as CybORG results. A passing simulator
smoke test is not a benchmark experiment and is not evidence of superiority.
