#!/usr/bin/env bash
# Matched-budget ablations for RA-HMAPPO. Requires the separately trained risk checkpoint.
set -euo pipefail

PYTHON="${PYTHON:-python}"
STEPS="${STEPS:-1000000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-256}"
DEVICE="${DEVICE:-cpu}"
EPISODE_STEPS="${EPISODE_STEPS:-500}"
RISK_CHECKPOINT="${RISK_CHECKPOINT:-artifacts/standard_cc4/cc4_risk.pt}"
OUT="${OUT:-artifacts/cc4_ablations}"
RESULTS="${RESULTS:-results/cc4_ablations}"
TRAIN_SEEDS=(11 22 33 44 55 66 77 88 99 111)
EVAL_SEEDS=($(seq 501 600))

if [[ ! -f "$RISK_CHECKPOINT" ]]; then
  echo "Missing risk checkpoint: $RISK_CHECKPOINT. Run the standard matrix risk-data stage first." >&2
  exit 2
fi

mkdir -p "$OUT" "$RESULTS"

run_variant() {
  local variant="$1" risk_mode="$2" mapping="$3" aggregation="$4"
  local risk_args=()
  if [[ "$risk_mode" == "predicted" ]]; then
    risk_args=(--risk-checkpoint "$RISK_CHECKPOINT")
  fi
  for seed in "${TRAIN_SEEDS[@]}"; do
    "$PYTHON" scripts/train_ra_hmappo_cc4.py "${risk_args[@]}" \
      --risk-mode "$risk_mode" --task-mapping "$mapping" \
      --reward-aggregation "$aggregation" --steps "$STEPS" \
      --rollout-steps "$ROLLOUT_STEPS" --episode-steps "$EPISODE_STEPS" \
      --device "$DEVICE" --seed "$seed" \
      --output "$OUT/${variant}_seed${seed}.pt"
  done
  for seed in "${TRAIN_SEEDS[@]}"; do
    local eval_risk_args=()
    if [[ "$risk_mode" == "predicted" ]]; then
      eval_risk_args=(--risk-checkpoint "$RISK_CHECKPOINT")
    fi
    "$PYTHON" scripts/evaluate_ra_hmappo_cc4.py \
      --checkpoint "$OUT/${variant}_seed${seed}.pt" "${eval_risk_args[@]}" \
      --seeds "${EVAL_SEEDS[@]}" --episode-steps "$EPISODE_STEPS" \
      --device "$DEVICE" --profile-inference --output "$RESULTS/${variant}_seed${seed}.jsonl"
  done
}

# Full model and one-factor-at-a-time ablations.
run_variant full predicted semantic sum
run_variant no_risk constant semantic sum
run_variant no_semantic_mapping predicted unconstrained sum
run_variant mean_team_reward predicted semantic mean

"$PYTHON" scripts/analyze_cc4_ablations.py \
  --full-glob "$RESULTS/full_seed*.jsonl" \
  --no-risk-glob "$RESULTS/no_risk_seed*.jsonl" \
  --no-mapping-glob "$RESULTS/no_semantic_mapping_seed*.jsonl" \
  --mean-reward-glob "$RESULTS/mean_team_reward_seed*.jsonl" \
  --output "$RESULTS/ablation_analysis.json"

echo "Ablation matrix complete. Inspect raw records, confidence intervals and diagnostics."
