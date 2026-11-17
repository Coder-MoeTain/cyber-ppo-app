# CC4 risk-estimator data and calibration pipeline

The hierarchical policy requires a scalar local risk input for each Blue agent.
This pipeline trains a supervised estimator from the agent's local
`BlueFlatWrapper` observation. The data collector reads simulator ground
truth only to create offline labels: an agent-zone label is positive when a
red session is present on any host assigned to that Blue agent. The ground
truth is not passed to the policy or estimator at inference.

## Collect disjoint datasets

Use different seed sets for training and validation; never overlap them.

```bash
python scripts/collect_cc4_risk_dataset.py --seeds 100 101 102 103 104 --steps 100 --output data/risk_train.npz
python scripts/collect_cc4_risk_dataset.py --seeds 201 202 203 --steps 100 --output data/risk_validation.npz
python scripts/train_risk_model.py --train-data data/risk_train.npz --validation-data data/risk_validation.npz --output artifacts/cc4_risk.pt
```

The collection script uses Sleep for Blue actions so that the policy does not
contaminate its own labels through learned interventions. This creates a
specific observational data distribution; later studies should assess
distribution shift under active defense policies.

## Calibration and metrics

The trainer uses a weighted binary cross-entropy objective, episode-seed
separation, validation-loss early stopping and temperature scaling. It reports
AUROC, AUPRC, Brier score, expected calibration error and positive prevalence.
Inspect the class balance and validation metrics before using the model.
AUROC/AUPRC are undefined for a validation set containing only one class.

## Critical caveats

- A red session anywhere in an agent's local zone is a coarse binary target,
  not a complete probability of attack success or asset compromise.
- The risk label definition and host-to-agent mapping must be verified against
  the pinned simulator. Audit the label extraction before trusting results.
- Risk-model validation must use held-out episodes and scenario shifts.
- Do not pass privileged state, true red locations, or labels to the runtime
  predictor.
- This is a baseline estimator pipeline, not yet a validated calibrated
  threat-probability model. Publish its limitations and calibration results.
