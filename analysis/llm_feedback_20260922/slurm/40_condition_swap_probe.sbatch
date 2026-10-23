#!/usr/bin/env bash
#SBATCH --job-name=llmtd-condswap
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${SWAP_REP:?missing representation}"
: "${SWAP_ADAPTER:?missing adapter checkpoint}"
: "${SWAP_SEED:?missing seed}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/condition_swap_probe.py" \
  --repo "$REPO" \
  --config "$RUN_ROOT/full_train_1028011/config.pkl" \
  --checkpoint "$RUN_ROOT/full_train_1028011/ema_model_epoch_8000.pt" \
  --adapter "$SWAP_ADAPTER" \
  --output "$RUN_ROOT/condition_swap_${SWAP_REP}_${SWAP_SEED}_${SLURM_JOB_ID}" \
  --samples "${SWAP_SAMPLES:-512}" \
  --steps "${SWAP_STEPS:-30}" \
  --guidance "${SWAP_GUIDANCE:-2}" \
  --pair-set "${SWAP_PAIR_SET:-heldout}" \
  --seed "$SWAP_SEED"
