#!/usr/bin/env bash
# Fidelity-focused screen on the strongest prior Shoppers configurations.
# Restores only numerical graph endpoints and compares against the matched
# final-repair output under the same operator.
#SBATCH --job-name=llmtd-shop-marginal
#SBATCH --partition=prioritized
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
i=$SLURM_ARRAY_TASK_ID
profile_index=$((i / 3))
strength_index=$((i % 3))
if [[ "$profile_index" == 0 ]]; then
  PROFILE=weak
  SOURCE="$A/outputs/e2e_guidance_recal_pretrained_weak_a4_1062341_1"
else
  PROFILE=moderate
  SOURCE="$A/outputs/e2e_guidance_recal_pretrained_moderate_a4_1062341_3"
fi
case "$strength_index" in
  0) LABEL=r010; STRENGTH=.10;;
  1) LABEL=r025; STRENGTH=.25;;
  2) LABEL=r050; STRENGTH=.50;;
esac
OUT="$A/outputs/shoppers_partial_restore_${SLURM_ARRAY_JOB_ID}_${i}_${PROFILE}_${LABEL}"
mkdir -p "$OUT/samples"

cd "$REPO"
"$PY" "$A/scripts/paired_marginal_restoration.py" \
  --reference "$SOURCE/samples/unconditional.csv" \
  --guided "$SOURCE/samples/x0_reward_calibrated.csv" \
  --graph "$SOURCE/active_graph.json" --strength "$STRENGTH" \
  --output "$OUT/samples/restored_dynamic.csv"
"$PY" "$A/scripts/paired_marginal_restoration.py" \
  --reference "$SOURCE/samples/unconditional.csv" \
  --guided "$SOURCE/samples/final_reward_calibrated.csv" \
  --graph "$SOURCE/active_graph.json" --strength "$STRENGTH" \
  --output "$OUT/samples/restored_repair.csv"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset shoppers --graph "$SOURCE/evaluation_graph.json" \
  --sample "tabdiff=$SOURCE/samples/unconditional.csv" \
  --sample "final_repair=$SOURCE/samples/final_reward_calibrated.csv" \
  --sample "dynamic=$SOURCE/samples/x0_reward_calibrated.csv" \
  --sample "restored_final_repair=$OUT/samples/restored_repair.csv" \
  --sample "restored_dynamic=$OUT/samples/restored_dynamic.csv" \
  --seed 20261301 --output "$OUT/eval"
