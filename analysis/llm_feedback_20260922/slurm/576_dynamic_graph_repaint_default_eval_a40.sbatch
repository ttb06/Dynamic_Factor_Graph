#!/usr/bin/env bash
#SBATCH --job-name=llmtd-dg-repaint-def-e
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=8-11
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SAMPLE_JOB_ID=1070752
labels=(mid25_broad mid40_broad final_broad actionable_mid25)
i=$SLURM_ARRAY_TASK_ID
label=${labels[$((i - 8))]}
trigger=mid
if [[ "$i" -eq 10 ]]; then trigger=final; fi
sample_dir="$A/outputs/dg_repaint_default_${label}_${SAMPLE_JOB_ID}_${i}/samples"
method="graph_repaint_${trigger}"

cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" \
  --sample "tabdiff=$sample_dir/tabdiff.csv" \
  --sample "final_repair=$sample_dir/calibrated_final_repair.csv" \
  --sample "${method}=$sample_dir/${method}.csv" \
  --sample "${method}_plus_repair=$sample_dir/${method}_plus_repair.csv" \
  --seed 20261440 \
  --output "$A/outputs/dg_repaint_eval_default_${label}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
