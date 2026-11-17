# Team UC-inspired CC4 heuristic baseline

The primary `heuristic` baseline is now a stateful Team UC-inspired policy
implemented in `CC4TeamUCInspiredPolicy`. Its action order is:

1. React to local connection-event flags with Restore.
2. React to process-event flags with Remove, falling back to Restore.
3. Reconcile the observed subnet blocking vector with the communication-policy
   vector using valid BlockTrafficZone/AllowTrafficZone actions.
4. During mission phases A/B, prefer Monitor.
5. During preplanning, round-robin through valid Analyse actions and then
   DeployDecoy actions; Sleep is the final fallback.

All actions are constrained by the BlueFixedActionWrapper action mask. The
policy uses the fixed CC4 BlueFlatWrapper vector layout and the published
CC4 subnet ordering. The separate `alert-only` baseline remains available as
a simple Analyse-on-alert comparator.

## Reproduction limits

This is an independent reimplementation inspired by the published Team UC
description, not the original competition source. The standard BlueFlatWrapper
vector used here does not expose the malicious-file flags described in the
published heuristic, and some decision details are simplified. Do not call it
an exact Team UC reproduction or claim the published score. Verify the action
semantics on the pinned CC4 release and report this difference.

The heuristic must be evaluated on exactly the same held-out seeds and episode
horizon as IPPO, MAPPO and RA-HMAPPO. A paper should include both this
Team UC-inspired comparator and the simpler alert-only baseline, and disclose
any deviations from the published policy.
