#!/usr/bin/env bash
# Full standard-CC4 experiment matrix. Long-running; reserve compute before use.
set -euo pipefail

PYTHON="${PYTHON:-python}"
STEPS="${STEPS:-1000000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-256}"
DEVICE="${DEVICE:-cpu}"
EPISODE_STEPS="${EPISODE_STEPS:-500}"
OUT="${OUT:-artifacts/standard_cc4}"
RESULTS="${RESULTS:-results/standard_cc4}"
TRAIN_SEEDS=(11 22 33 44 55 66 77 88 99 111)
RISK_TRAIN_SEEDS=($(seq 121 140))
RISK_VALIDATION_SEEDS=($(seq 201 220))
RISK_CALIBRATION_SEEDS=($(seq 301 320))
RISK_TEST_SEEDS=($(seq 401 420))
EVAL_SEEDS=($(seq 501 600))

mkdir -p "$OUT" "$RESULTS" data
"$PYTHON" scripts/collect_cc4_risk_dataset.py --seeds "${RISK_TRAIN_SEEDS[@]}" \
  --steps "$EPISODE_STEPS" --output data/cc4_risk_train.npz
"$PYTHON" scripts/collect_cc4_risk_dataset.py --seeds "${RISK_VALIDATION_SEEDS[@]}" \
  --steps "$EPISODE_STEPS" --output data/cc4_risk_validation.npz
"$PYTHON" scripts/collect_cc4_risk_dataset.py --seeds "${RISK_CALIBRATION_SEEDS[@]}" \
  --steps "$EPISODE_STEPS" --output data/cc4_risk_calibration.npz
"$PYTHON" scripts/collect_cc4_risk_dataset.py --seeds "${RISK_TEST_SEEDS[@]}" \
  --steps "$EPISODE_STEPS" --output data/cc4_risk_test.npz
"$PYTHON" scripts/train_risk_model.py --train-data data/cc4_risk_train.npz \
  --validation-data data/cc4_risk_validation.npz \
  --calibration-data data/cc4_risk_calibration.npz \
  --test-data data/cc4_risk_test.npz --device "$DEVICE" --output "$OUT/cc4_risk.pt"

for seed in "${TRAIN_SEEDS[@]}"; do
  "$PYTHON" scripts/train_ra_hmappo_cc4.py --risk-checkpoint "$OUT/cc4_risk.pt" \
    --steps "$STEPS" --rollout-steps "$ROLLOUT_STEPS" --episode-steps "$EPISODE_STEPS" \
    --device "$DEVICE" --seed "$seed" --output "$OUT/ra_hmappo_seed${seed}.pt"
  "$PYTHON" scripts/train_ippo_cc4.py --steps "$STEPS" --rollout-steps "$ROLLOUT_STEPS" \
    --episode-steps "$EPISODE_STEPS" --device "$DEVICE" --seed "$seed" --output "$OUT/ippo_seed${seed}.pt"
  "$PYTHON" scripts/train_mappo_cc4.py --steps "$STEPS" --rollout-steps "$ROLLOUT_STEPS" \
    --episode-steps "$EPISODE_STEPS" --device "$DEVICE" --seed "$seed" --output "$OUT/mappo_seed${seed}.pt"
done

"$PYTHON" scripts/evaluate_baseline_cc4.py --baseline heuristic \
  --seeds "${EVAL_SEEDS[@]}" --episode-steps "$EPISODE_STEPS" --device "$DEVICE" --profile-inference \
  --output "$RESULTS/heuristic.jsonl"

"$PYTHON" scripts/evaluate_baseline_cc4.py --baseline alert-only \
  --seeds "${EVAL_SEEDS[@]}" --episode-steps "$EPISODE_STEPS" --device "$DEVICE" --profile-inference \
  --output "$RESULTS/alert_only.jsonl"

for seed in "${TRAIN_SEEDS[@]}"; do
  "$PYTHON" scripts/evaluate_ra_hmappo_cc4.py --checkpoint "$OUT/ra_hmappo_seed${seed}.pt" \
    --risk-checkpoint "$OUT/cc4_risk.pt" --seeds "${EVAL_SEEDS[@]}" \
    --episode-steps "$EPISODE_STEPS" --device "$DEVICE" --profile-inference --output "$RESULTS/ra_hmappo_seed${seed}.jsonl"
  "$PYTHON" scripts/evaluate_baseline_cc4.py --baseline ippo \
    --checkpoint "$OUT/ippo_seed${seed}.pt" --seeds "${EVAL_SEEDS[@]}" \
    --episode-steps "$EPISODE_STEPS" --device "$DEVICE" --profile-inference --output "$RESULTS/ippo_seed${seed}.jsonl"
  "$PYTHON" scripts/evaluate_baseline_cc4.py --baseline mappo \
    --checkpoint "$OUT/mappo_seed${seed}.pt" --seeds "${EVAL_SEEDS[@]}" \
    --episode-steps "$EPISODE_STEPS" --device "$DEVICE" --profile-inference --output "$RESULTS/mappo_seed${seed}.jsonl"
done

"$PYTHON" scripts/analyze_cc4_results.py \
  --ra-glob "$RESULTS/ra_hmappo_seed*.jsonl" \
  --ippo-glob "$RESULTS/ippo_seed*.jsonl" \
  --mappo-glob "$RESULTS/mappo_seed*.jsonl" \
  --heuristic-file "$RESULTS/heuristic.jsonl" \
  --output "$RESULTS/paired_analysis.json"

"$PYTHON" scripts/artifact_manifest.py create \
  --paths "$OUT" "$RESULTS" data/cc4_risk_train.npz data/cc4_risk_validation.npz \
  data/cc4_risk_calibration.npz data/cc4_risk_test.npz \
  --output "$RESULTS/artifact_manifest.json"

echo "Standard CC4 matrix and paired analysis complete. Review diagnostics before interpreting results."
