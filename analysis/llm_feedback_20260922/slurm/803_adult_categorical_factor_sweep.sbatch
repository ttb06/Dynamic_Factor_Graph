#!/usr/bin/env bash
# Adult is the strongest categorical-dependency test in the current datasets.
# This isolates train-only categorical factor messages from numerical repair.
#SBATCH --job-name=llmtd-adult-catfactor
#SBATCH --partition=prioritized
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
i=$SLURM_ARRAY_TASK_ID

case "$i" in
  0) LABEL=target_s010_pair2; CAT=(--categorical-guidance-mode target --categorical-guidance-strength .10 --categorical-guidance-pair-scale 2);;
  1) LABEL=target_s025_pair2; CAT=(--categorical-guidance-mode target --categorical-guidance-strength .25 --categorical-guidance-pair-scale 2);;
  2) LABEL=target_s050_pair2; CAT=(--categorical-guidance-mode target --categorical-guidance-strength .50 --categorical-guidance-pair-scale 2);;
  3) LABEL=marginal_s010; CAT=(--categorical-guidance-mode marginal --categorical-guidance-strength .10);;
  4) LABEL=hybrid_s025_scaled; CAT=(--categorical-guidance-mode marginal_target --categorical-guidance-strength .25 --categorical-guidance-marginal-scale .10 --categorical-guidance-pair-scale 2);;
  5) LABEL=pairwise_s025_scaled; CAT=(--categorical-guidance-mode marginal_pairwise --categorical-guidance-strength .25 --categorical-guidance-marginal-scale .10 --categorical-guidance-pair-scale 2 --categorical-guidance-max-pairs 16);;
esac

GRAPH="$A/outputs/extended_bscg_adult_1081004/graph.json"
OUT="$A/outputs/adult_categorical_factors_${LABEL}_${SLURM_ARRAY_JOB_ID}_${i}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset adult --graph "$GRAPH" \
  --include-edge-ids bscg_001,bscg_003 \
  --config "$A/outputs/extended_full_train_adult_1081005/config.pkl" \
  --checkpoint "$A/outputs/extended_full_train_adult_1081005/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261140 \
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .02 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --projection-strength 0 --projection-min-t 0 --projection-max-t 0 \
  --distribution-transport-mode none \
  --categorical-guidance-min-t .20 --categorical-guidance-max-t .50 \
  --categorical-guidance-schedule middle_peak "${CAT[@]}" \
  --stochastic-sampler --second-order-correction \
  --output "$OUT/sample"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset adult --graph "$GRAPH" \
  --sample "TabDiff=$OUT/sample/samples/unconditional.csv" \
  --sample "CategoricalGraph=$OUT/sample/samples/dense_factor.csv" \
  --seed 20261140 --max-rows 4096 --output "$OUT/eval"
