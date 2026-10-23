#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-train
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
RUN="$ROOT/baseline_runs/harpoon_general_adult_20261002"
PY="$ROOT/baseline_runs/envs/harpoon-official/bin/python"
cd "$RUN"
"$PY" train_repaint.py --dataname adult --gpu 0 --epochs 1000 --timesteps 200 --batch_size 1024 --hid_dim 1024
