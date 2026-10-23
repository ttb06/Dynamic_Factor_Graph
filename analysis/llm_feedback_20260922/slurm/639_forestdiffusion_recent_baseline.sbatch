#!/usr/bin/env bash
#SBATCH --job-name=llmtd-forest24
#SBATCH --partition=prioritized
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(default shoppers default shoppers)
duplicate_ks=(20 20 100 100)
labels=(screen screen official official)
i=$SLURM_ARRAY_TASK_ID
dataset=${datasets[$i]}
k=${duplicate_ks[$i]}
label=${labels[$i]}
output="$A/outputs/forestdiffusion_${dataset}_${label}_k${k}_${SLURM_ARRAY_JOB_ID}_${i}"
cd "$REPO"
export PYTHONPATH="$A/optional_dependency_shims:$ROOT/baseline_repos/forestdiffusion/Python-Package/base-ForestDiffusion:${PYTHONPATH:-}"
"$PY" "$A/scripts/run_forestdiffusion_baseline.py" \
  --repo "$REPO" --forest-repo "$ROOT/baseline_repos/forestdiffusion" \
  --dataset "$dataset" --duplicate-k "$k" --n-t 50 --n-estimators 100 \
  --n-jobs 1 --n-batch 10 --gpu-hist --samples 4096 --seed 20261015 --output "$output"
