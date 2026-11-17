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
- [x] Offline-supervised risk-estimation pipeline with privileged labels isolated
  to offline dataset creation and local-observation inference.
- [x] Risk-model fitting, model-selection, temperature-calibration and final-test
  seed partitions are now separate; split overlap is rejected.
- [x] Final risk-model test metrics are saved separately from model-selection
  and calibration metrics, with episode-clustered confidence intervals.
- [x] CI workflow definition added for Python 3.10/3.11 unit, regression and compile checks; successful CI execution remains pending.
- [x] Analysis rejects incomplete default matrices (10 training seeds x 100
  held-out evaluation seeds), duplicate pairs and mismatched method seeds.
- [x] Matched-budget ablation runner and paired analysis added for risk removal,
  semantic task-mask removal and sum-versus-mean reward aggregation.
- [x] Team UC-inspired stateful, action-mask-aware heuristic and simple
  alert-only comparator; neither is an exact competition-source reproduction.
- [ ] Validate the Team UC-inspired implementation against published policy
  details and report its missing malicious-file features.
- [x] Official CAGE Challenge 4 source revision pinned in `configs/cc4.lock.yaml`.
- [x] Real CC4 adapter reset, same-seed replay, action-mask and one-step smoke validation.
- [x] Real CC4 offline risk-dataset collection smoke and RA-HMAPPO rollout/update smoke.
- [ ] Pass the official upstream CC4 test suite.
- [ ] Audit privileged risk-label semantics and independently review test-set calibration.
- [ ] Empirically validate task ontology and ablate alternative option mappings.
- [ ] Validate reward aggregation and episode-boundary semantics on real CC4.
- [ ] Complete matched-budget training and evaluation for heuristic, IPPO,
  MAPPO and RA-HMAPPO.

## Experimental and manuscript

- [x] Regression tests for paired result matrices, duplicate/missing seed pairs,
  heuristic pairing, deterministic bootstrap, and risk-split separation added;
  execution in CI is still pending.
- [x] Manuscript structure drafted with explicit evidence-pending sections; it
  is not a complete submission manuscript until results, related-work citations,
  discussion, and conclusions are completed.
- [x] Experiment execution and artifact-preservation guide added.

- [ ] Freeze hypotheses, primary endpoints, scenario configs and exclusions.
- [ ] Execute the full risk-data collection and report final test metrics.
- [ ] Run at least 10 independent training seeds.
- [ ] Evaluate at least 100 episodes per method and scenario, including
  predeclared generalization shifts.
- [ ] Report team/per-agent reward, red-session outcomes, availability,
  invalid-action rate, unnecessary intervention, latency and overhead.
- [ ] Execute the risk-input, task-mapping and reward-aggregation ablation matrix.
- [ ] Add and execute deterministic task-allocation and risk-calibration/uncertainty ablations.
- [ ] Archive raw episode records, configs, checkpoints, source provenance,
  statistical scripts and environment logs.
- [ ] Complete novelty review, manuscript, verified citations, artifact license
  and clean-machine reproduction.

Do not present mock-environment returns as CybORG results. A passing simulator
smoke test is not a benchmark experiment and is not evidence of superiority.
