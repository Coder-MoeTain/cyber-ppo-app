# Phase 1: Environment and Interface Validation

1. Review CybORG, CAGE enterprise scenarios, MAPPO, hierarchical RL, and MARL cyber defense literature.
2. Pin the CybORG commit and record dependency versions.
3. Run the official enterprise scenario tutorial unchanged.
4. Enumerate Blue agents, observation schemas, action labels, invalid-action behavior, and reward semantics.
5. Map logical roles to actual environment actors.
6. Document local observations and permitted coordination messages.
7. Ensure hidden simulator ground truth is not leaked into decentralized policies.
8. Test reset determinism, action validity, observation shapes, and reward accounting.

## Acceptance criteria
- Pinned, reproducible CybORG setup.
- Official scenario instantiates and resets.
- Runtime-derived agent/action/observation documentation.
- No hidden-state leakage.
- Mock tests pass.
- Research questions and primary metrics finalized.

The mock environment validates software plumbing only; it is not a CybORG substitute or benchmark.
