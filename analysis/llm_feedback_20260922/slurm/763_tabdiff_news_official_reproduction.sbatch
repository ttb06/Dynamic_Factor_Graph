#!/usr/bin/env bash
# Untouched official TabDiff training entrypoint and its 20-table report.  A
# dedicated experiment name prevents overwriting all existing checkpoints.
#SBATCH --job-name=llmtd-tabdiff-news-repro
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
PY="$ROOT/baseline_runs/envs/tabdiff/bin/python"
EXP=codex_repro_20261018_fast
[[ ! -e "$REPO/tabdiff/ckpt/news/$EXP" ]]
cd "$REPO"
"$PY" main.py --dataname news --mode train --gpu 0 \
  --exp_name "$EXP" --no_wandb
"$PY" main.py --dataname news --mode test --gpu 0 \
  --exp_name "$EXP" --report --num_runs 20 --no_wandb
