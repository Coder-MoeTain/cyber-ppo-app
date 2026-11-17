# Risk-aware hierarchical policy

For agent i, the decentralized manager samples a task z_i from a policy
conditioned on its local observation o_i and normalized risk r_i. A decentralized
worker samples an environment action conditioned on o_i, r_i and z_i. A
centralized critic consumes the concatenated global state during training.
Managers and workers do not require global state during execution.

The module exposes task/action distributions, sampling, log probabilities,
entropies, and evaluation methods suitable for storing hierarchical rollouts.
The task ontology and mapping to operational CC4 actions must be defined and
validated by the experiment before training.

## Important limitation

This is a trainable policy architecture, not a completed RA-HMAPPO algorithm.
The current MAPPO trainer does not yet collect manager decisions or optimize
both policy levels. A hierarchical rollout buffer, joint or separately weighted
clipped PPO objectives, manager/worker credit assignment, and live CybORG
evaluation remain necessary. Unit tests validate tensor shapes, masks and
log-probability consistency only; they are not benchmark results.
