#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-train3
#SBATCH --partition=prioritized
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
ENV="$ROOT/baseline_runs/envs/harpoon-official"
RUN_ROOT="$ROOT/baseline_runs/harpoon_official_multidata_20261008"
DATASETS=(adult default shoppers)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}

NVLIB=$(find "$ENV/lib/python3.11/site-packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)
export LD_LIBRARY_PATH="$ROOT/baseline_runs/envs/grdm/lib:$NVLIB:${LD_LIBRARY_PATH:-}"
cd "$RUN_ROOT/$dataset"
"$ENV/bin/python" train_repaint.py --dataname "$dataset" --gpu 0 \
  --epochs 1000 --timesteps 200 --batch_size 1024 --hid_dim 1024
test -s "saved_models/$dataset/diffputer_selfmade.pt"
