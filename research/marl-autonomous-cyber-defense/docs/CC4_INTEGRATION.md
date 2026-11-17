# CAGE Challenge 4 integration

The adapter follows the documented CC4 API: `CybORG(scenario_generator=EnterpriseScenarioGenerator())`, wrapped by `BlueFlatWrapper(env=cyborg, pad_spaces=True)`, with five external Blue agents (`blue_agent_0` through `blue_agent_4`). The wrapper exposes per-agent reset/step observations, action spaces, labels and dynamic action masks.

References:
- https://cage-challenge.github.io/cage-challenge-4/pages/tutorials/01_Getting_Started/2_Getting_Started/
- https://cage-challenge.github.io/cage-challenge-4/pages/reference/wrappers/BlueFixedActionWrapper/
- https://cage-challenge.github.io/cage-challenge-4/pages/reference/wrappers/BlueFlatWrapper/

## Validation
Install the official CybORG release at a recorded commit in an isolated environment, then run `pip install -e ".[dev]"`, `python scripts/validate_cyborg.py`, and optionally `python scripts/validate_cyborg.py --one-step`.

Archive exact Git SHA, package versions, OS, Python version, scenario parameters and seed behavior. Import success alone is not simulator validation.

## Action masks and reproducibility
Masks are episode-specific. Apply them before sampling and retain them in rollouts for consistent log-probability recomputation. The adapter seed argument does not yet guarantee simulator seeding; verify the pinned release's API and document deterministic replay limits.
