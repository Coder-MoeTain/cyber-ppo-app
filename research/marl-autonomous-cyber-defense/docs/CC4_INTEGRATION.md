# CAGE Challenge 4 integration

## Pinned upstream source

The benchmark source is pinned to
`cage-challenge/cage-challenge-4@8c3c50ca54b176c2de199847944e8dcc035497e3`
(CybORG version 4.0). See `configs/cc4.lock.yaml` and
`requirements-cc4-validation.txt`. Do not silently track upstream main.

The adapter constructs `CybORG(scenario_generator=EnterpriseScenarioGenerator())`
and wraps it with `BlueFlatWrapper(env=cyborg, pad_spaces=True)`. The five
external Blue agents are `blue_agent_0` through `blue_agent_4`. BlueFlatWrapper
provides fixed-order vector observations; BlueFixedActionWrapper provides a
fixed integer action mapping and episode-specific action masks. The action
mask must be stored in the rollout and reused during PPO log-probability
recomputation.

## Validation commands

With the pinned upstream source checked out under `vendor/CybORG`:

```bash
python -m pip install -r requirements-cc4-validation.txt
python -m pip install --no-deps -e vendor/CybORG
python -m pip install --no-deps -e .
python -m pytest -q vendor/CybORG/CybORG/Tests/test_cc4
python scripts/validate_cyborg.py --one-step --verify-seed
```

The official wrapper package imports its RLlib wrapper from the package
initializer, so the validation dependency set includes a pinned Ray/RLlib
version even though this project uses BlueFlatWrapper directly.

The integration workflow performs these checks in GitHub Actions. A passing
smoke test confirms interface compatibility for the pinned version; it does
not validate training convergence or benchmark effectiveness.

## Reproducibility and known semantics

- Reset seeds are forwarded to the underlying CybORG reset API.
- The wrapper returns per-agent reward streams; the hierarchical trainer's
  default team objective is their sum. Always report individual agent rewards
  and justify the aggregation choice.
- Validate termination/truncation semantics against the pinned implementation.
- Record source SHA, Python/dependency versions, OS, scenario steps, seed and
  action-mask dimensions with every run.
- The policy must not access privileged simulator state. The offline risk
  dataset collector uses red-session state only for supervised labels; the
  deployed risk predictor consumes the Blue agent's observation vector only.

## Official references

- https://github.com/cage-challenge/cage-challenge-4
- https://cage-challenge.github.io/cage-challenge-4/pages/tutorials/01_Getting_Started/2_Getting_Started/
- https://cage-challenge.github.io/cage-challenge-4/pages/reference/wrappers/BlueFixedActionWrapper/
- https://cage-challenge.github.io/cage-challenge-4/pages/reference/wrappers/BlueFlatWrapper/
