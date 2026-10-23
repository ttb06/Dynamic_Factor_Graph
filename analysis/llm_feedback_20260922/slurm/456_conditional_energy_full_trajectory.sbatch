#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cond-energy-full
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-8
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
i=$SLURM_ARRAY_TASK_ID
request_i=$((i / 3))
gate_i=$((i % 3))
requests=(range category both)
gates=(dynamic frozen global)
request=${requests[$request_i]}
gate=${gates[$gate_i]}
case "$request" in
  range) strengths="0.8,8" ;;
  category) strengths="0.4,64" ;;
  both) strengths="0.8,64" ;;
esac

"$PY" "$A/scripts/harpoon_tabdiff_mixed_proxy.py" --repo "$REPO" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --request "$request" --samples 4096 --steps 50 --seed 20261376 \
  --start-t 1 --active-min-t 0 --condition-column MARRIAGE --condition-value 1 \
  --gate-mode "$gate" --strengths "$strengths" \
  --output "$A/outputs/conditional_energy_${request}_${gate}_${SLURM_ARRAY_JOB_ID}_${i}"
