# Architecture Notes

## Roles
Detection investigates suspicious observations; risk estimates asset-level risk from permitted features; response selects constrained defensive recommendations. Recovery is a later extension.

## Hierarchy
Phase 1 has a deterministic risk-priority coordinator as an interpretable baseline. The research contribution requires a trainable high-level policy assigning tasks or conditioning low-level policies. A deterministic coordinator alone is not hierarchical RL.

## CTDE
A centralized critic may consume permitted joint training state. At execution, each actor uses only its local observation and explicitly allowed coordination messages.

## Safety
Action mapping must enforce environment permissions and valid action masks. Intended for simulation and isolated authorized cyber ranges.
