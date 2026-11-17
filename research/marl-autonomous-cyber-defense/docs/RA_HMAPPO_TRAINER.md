# RA-HMAPPO trainer

The hierarchical trainer optimizes a manager and worker policy for each
decentralized agent. At each decision, the manager samples a task from
\(\pi_H(z_i\mid o_i,r_i)\); the worker samples an environment action from
\(\pi_L(a_i\mid o_i,r_i,z_i)\). The centralized critic estimates team value
from the concatenated global state. PPO clipping is applied per agent to the
joint manager-plus-worker log-probability ratio; manager and worker entropy
terms are reported separately.

## Operational task masks

The CC4 runner supplies `CC4TaskActionMapper`. Its six options map to
Monitor/Sleep, Analyse, Remove, Restore/AllowTrafficZone, BlockTrafficZone,
and DeployDecoy action families. The manager mask enables only options with
at least one currently valid action; the worker mask is the intersection of
the selected task's action family and the BlueFixedActionWrapper action mask.
Both masks are stored in the rollout and reused during PPO updates.

This mapping is a declared algorithmic inductive bias and must be ablated.
It does not imply that the task ontology is optimal.

## Risk interface

A caller must supply `risk_fn(agent, observation, info) -> float`, with a
validated value in [0,1]. The offline risk pipeline trains on local Blue
observations and uses privileged red-session state only to create training
labels. The runtime predictor receives only the acting agent's observation.
Report calibration and test under scenario shifts before relying on the score.

## Reward aggregation

The default team reward is the sum of the five per-agent rewards, representing
cooperative total utility. `reward_aggregation="mean"` is available for
sensitivity analysis. Papers must report per-agent rewards and justify the
chosen team objective.

## Scope and limitations

The trainer supports synchronous episodes where all agents terminate or
truncate together. It rejects partial termination rather than silently
mis-handling a centralized state. Truncation bootstraps from the final
observation, while true termination does not. Subsequent episodes use
deterministic seed offsets.

The trainer and task masks have mock tests; live CC4 integration and
benchmark convergence still require successful pinned-simulator validation.
No claim of benchmark effectiveness follows from unit tests.
