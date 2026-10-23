#!/usr/bin/env bash
#SBATCH --job-name=llmtd-subgraph-scope2
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="${ANALYSIS_RUN_ROOT:-$A/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SEEDS=(20260926 20260927)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT="$R/subgroup_scope_confirm_seed${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/subgroup_conditioned_sampling.py" \
  --repo "$REPO" \
  --config "$R/full_train_default_1033760/config.pkl" \
  --checkpoint "$R/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --adapter "$R/subgroup_amount_adapter_1050333/adapter.pt" \
  --output "$OUT" --samples 2048 --steps 30 --seed "$SEED" --guidances 1:2
