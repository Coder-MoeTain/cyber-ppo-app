# Publication readiness gates

**Current classification: NOT publication ready.** Mock returns and policy
architecture tests are not benchmark evidence.

## Software and method
- [x] Five-agent CC4 adapter boundary and action-mask interface.
- [x] CTDE MAPPO trainer and paired statistical utilities.
- [x] Risk-conditioned manager/worker policy architecture; unit-tested.
- [ ] Integrate manager and worker decisions into a hierarchical rollout buffer.
- [ ] Implement and test joint or separately weighted clipped PPO objectives.
- [ ] Define task ontology and safe mapping from high-level tasks to CC4 actions.
- [ ] Pin verified CybORG/CAGE commit and lock dependencies.
- [ ] Validate reset, terminal/truncation, masks, dimensions and seeding on live CC4.
- [ ] Implement Independent PPO and verify all baselines on the real simulator.
- [ ] Validate reward semantics and report per-agent and team outcomes.

## Experimental and manuscript
- [ ] Freeze hypotheses, primary endpoints and disjoint scenario splits.
- [ ] Run >=10 independent seeds where feasible and disclose deviations.
- [ ] Report attack success, critical-asset compromise, service availability,
  detection/mitigation latency, unnecessary defensive actions and overhead.
- [ ] Run risk, hierarchy, communication and reward ablations.
- [ ] Archive raw episode results, configs, checkpoints and environment provenance.
- [ ] Complete novelty review, manuscript, verified citations, artifact license
  and clean-machine reproduction.

Never present mock-environment results as CybORG results.
