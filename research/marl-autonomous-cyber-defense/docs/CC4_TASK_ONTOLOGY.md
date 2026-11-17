# CC4 hierarchical task ontology and action constraints

The manager's six discrete options are operationally grounded in CC4 action
families, rather than treated as arbitrary labels:

| Task ID | Option | Allowed action labels |
|---:|---|---|
| 0 | Monitor | Monitor, Sleep |
| 1 | Investigate | Analyse |
| 2 | Remediate | Remove, RemoveOtherSessions |
| 3 | Recover | Restore, AllowTrafficZone |
| 4 | Isolate | BlockTrafficZone |
| 5 | Deceive | DeployDecoy |

At each decision, the action mask supplied by BlueFixedActionWrapper is
intersected with the task's action-family mask. The manager's task mask is true
only when at least one valid action exists for that option. The worker then
samples only from the selected task's valid action subset. The selected
task-mask and worker-mask are stored in the rollout and reused for PPO
log-probability recomputation.

This prevents the manager from selecting an option that has no valid action
for the current agent/episode, and prevents the worker from executing an
action inconsistent with the selected option. The mapping is a declared
inductive bias; report it and ablate it. It is not evidence that the task
semantics are optimal.
