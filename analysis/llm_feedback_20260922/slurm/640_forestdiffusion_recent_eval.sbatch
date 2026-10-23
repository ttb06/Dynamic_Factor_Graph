#!/usr/bin/env bash
#SBATCH --job-name=llmtd-forest24-e
#SBATCH --partition=prioritized
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${FOREST_JOB_ID:?FOREST_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(default shoppers default shoppers)
duplicate_ks=(20 20 100 100)
labels=(screen screen official official)
graph_indices=(1 0 1 0)
i=$SLURM_ARRAY_TASK_ID
dataset=${datasets[$i]}; k=${duplicate_ks[$i]}; label=${labels[$i]}; gi=${graph_indices[$i]}
sample="$A/outputs/forestdiffusion_${dataset}_${label}_k${k}_${FOREST_JOB_ID}_${i}/forest_diffusion.csv"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/actionable_union_eval_${dataset}_1060955_${gi}/evaluation_graph.json" \
  --sample "forestdiffusion_${label}=$sample" --seed 20261015 \
  --output "$A/outputs/forestdiffusion_eval_${dataset}_${label}_${SLURM_ARRAY_JOB_ID}_${i}"
