#!/usr/bin/env bash
#SBATCH --job-name=llmtd-tabsyn24
#SBATCH --partition=prioritized
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
UPSTREAM="$ROOT/baseline_repos/tabsyn"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/lcats-official/bin/python"
datasets=(default shoppers)
i=$SLURM_ARRAY_TASK_ID
dataset=${datasets[$i]}
seed=$((20261015 + i))
output="$A/outputs/tabsyn_official_${dataset}_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$output"
cd "$UPSTREAM"
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$seed" --script "$UPSTREAM/main.py" \
  --dataname "$dataset" --method vae --mode train --gpu 0
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$seed" --script "$UPSTREAM/main.py" \
  --dataname "$dataset" --method tabsyn --mode train --gpu 0
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$seed" --script "$UPSTREAM/main.py" \
  --dataname "$dataset" --method tabsyn --mode sample --gpu 0 --steps 50 \
  --save_path "$output/tabsyn_full.csv"
"$PY" "$A/scripts/subsample_baseline_csv.py" --input "$output/tabsyn_full.csv" \
  --output "$output/tabsyn_4096.csv" --rows 4096 --seed "$seed"
