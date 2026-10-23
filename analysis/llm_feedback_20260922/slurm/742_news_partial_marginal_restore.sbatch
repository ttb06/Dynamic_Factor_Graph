#!/usr/bin/env bash
# Paired marginal-proximal screen.  Apply the identical restoration operator
# to dynamic guidance and final repair, then evaluate with the common metrics.
#SBATCH --job-name=llmtd-news-marginal
#SBATCH --partition=cpu
#SBATCH --array=0-11%12
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"

i=$SLURM_ARRAY_TASK_ID
seed_index=$((i / 4))
strength_index=$((i % 4))
case "$seed_index" in
  0) SEED=20261605; SOURCE="$A/outputs/news_auto_isotonic_1093899_3_s100_s20261605";;
  1) SEED=20261606; SOURCE="$A/outputs/news_auto_isotonic_1093899_4_s100_s20261606";;
  2) SEED=20261607; SOURCE="$A/outputs/news_auto_isotonic_1093899_5_s100_s20261607";;
esac
case "$strength_index" in
  0) LABEL=r010; STRENGTH=.10;;
  1) LABEL=r025; STRENGTH=.25;;
  2) LABEL=r050; STRENGTH=.50;;
  3) LABEL=r075; STRENGTH=.75;;
esac
OUT="$A/outputs/news_partial_restore_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}_s${SEED}"
mkdir -p "$OUT/samples"

cd "$REPO"
"$PY" "$A/scripts/paired_marginal_restoration.py" \
  --reference "$SOURCE/samples/unconditional.csv" \
  --guided "$SOURCE/samples/dense_factor.csv" --graph "$GRAPH" \
  --include-families llm_directional_bound --strength "$STRENGTH" \
  --output "$OUT/samples/restored_auto.csv"
"$PY" "$A/scripts/paired_marginal_restoration.py" \
  --reference "$SOURCE/samples/unconditional.csv" \
  --guided "$SOURCE/samples/final_repair.csv" --graph "$GRAPH" \
  --include-families llm_directional_bound --strength "$STRENGTH" \
  --output "$OUT/samples/restored_repair.csv"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$SOURCE/samples/unconditional.csv" \
  --sample "final_repair=$SOURCE/samples/final_repair.csv" \
  --sample "auto_isotonic=$SOURCE/samples/dense_factor.csv" \
  --sample "restored_final_repair=$OUT/samples/restored_repair.csv" \
  --sample "restored_auto=$OUT/samples/restored_auto.csv" \
  --seed "$SEED" --output "$OUT/eval"
