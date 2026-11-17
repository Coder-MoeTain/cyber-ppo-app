# CC4 rule-based baseline

`CC4AlertGuidedPolicy` is a transparent alert-guided baseline. It decodes the
documented BlueFlatWrapper subnet blocks, prioritizes the locally observed host
with process-creation/network-connection events, and selects a valid Analyse
action. If no local event is present, it chooses Sleep. Action masks are
respected.

This is an original simple reimplementation, not the original Team UC
competition policy and not a claim to reproduce published scores. The
published CC4 work describes more sophisticated event filtering and proactive
defense. The heuristic should be strengthened and validated before calling it
a strong baseline. Its purpose in this repository is to ensure that a
non-learning, mask-aware comparator exists and can be run under the same
evaluation seeds as IPPO, MAPPO and RA-HMAPPO.
