# Risk-aware hierarchical policy

For each agent i, the decentralized manager samples a task z_i from a policy
conditioned on local observation o_i and normalized risk r_i. A decentralized
worker samples an environment action conditioned on o_i, r_i and z_i. A
centralized critic consumes the concatenated global state during training;
manager and worker execution do not require global state.

The policy exposes manager and worker distributions, sampling, log
probabilities and entropies. The RA-HMAPPO trainer stores both decisions,
applies clipped PPO objectives to their joint log-probability ratio, and uses
a centralized team critic.

For CC4, the six task options are operationally grounded through
`CC4TaskActionMapper`: monitor, investigate, remediate, recover, isolate and
deceive. The manager sees only tasks with at least one valid action. The worker
is restricted to the selected task's action family intersected with the
simulator action mask.

## Scientific limitations

The option-to-action mapping is a design choice and must be ablated against
flat MAPPO and a latent/unconstrained hierarchy. The risk estimator is a
separate supervised model whose label validity, calibration and robustness
must be measured. The trainer rejects partial agent termination and assumes
synchronous team episodes. Mock tests validate implementation behavior only;
they do not establish real CC4 convergence, superiority or publication quality.
