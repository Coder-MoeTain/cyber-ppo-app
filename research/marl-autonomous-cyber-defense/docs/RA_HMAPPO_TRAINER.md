# RA-HMAPPO trainer

The hierarchical trainer optimizes a manager and worker policy for each
decentralized agent. At each decision, the manager samples a task from
\(\pi_H(z_i\mid o_i,r_i)\); the worker samples an environment action from
\(\pi_L(a_i\mid o_i,r_i,z_i)\). The centralized critic estimates team value
from the concatenated global state. PPO clipping is applied per agent to the
joint manager-plus-worker log probability ratio; manager and worker entropy
terms are reported separately.

## Risk interface

A caller must supply `risk_fn(agent, observation, info) -> float`, with a
validated value in [0,1]. This is intentional: CC4's fixed flat observations
are categorical/binary features, not a calibrated risk score by definition.
The risk estimator must be designed, trained, calibrated and evaluated
separately, with no privileged state leakage into decentralized execution.

## Reward aggregation

The default team reward is the sum of the five per-agent rewards, representing
cooperative total utility. `reward_aggregation="mean"` is available for
sensitivity analysis. Papers must report per-agent rewards and justify the
chosen team objective; do not silently average away zone-specific penalties.

## Scope and limitations

The trainer supports synchronous episodes where all agents terminate or
truncate together. It rejects partial termination rather than silently
mis-handling a centralized state. Truncation bootstraps from the final
observation, while true termination does not. Subsequent episodes use
deterministic seed offsets.

The code is an experimental implementation that must still be checked against
the pinned CC4 runtime. The task IDs are abstract and require a validated
operational ontology. Mock tests do not establish CC4 performance or research
effectiveness.
