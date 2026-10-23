#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-match
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
requests=(range category both)
request=${requests[$SLURM_ARRAY_TASK_ID]}
cd "$REPO"
"$PY" "$A/scripts/harpoon_tabdiff_mixed_proxy.py" --repo "$REPO" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --request "$request" --samples 4096 --steps 200 --seed 20261012 \
  --start-t 1 --active-min-t 0 --gate-mode global \
  --strengths '0.05,0.05:0.2,0.2:0.2,1:0.2,2:0.5,4' \
  --output "$A/outputs/harpoon_matched_${request}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
