#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cuts-common
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/cuts_common_tabdiff_baselines.py" --repo "$REPO" --config "$A/outputs/full_train_1028011/config.pkl" --checkpoint "$A/outputs/full_train_1028011/ema_model_epoch_8000.pt" --output "$A/outputs/cuts_common_tabdiff_${SLURM_JOB_ID}" --samples 4096 --best-of 8 --steps 30 --seed 20260930
