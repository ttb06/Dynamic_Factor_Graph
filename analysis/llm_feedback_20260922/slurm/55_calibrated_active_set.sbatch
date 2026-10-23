#!/usr/bin/env bash
#SBATCH --job-name=llmtd-calgate
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CAL_ADAPTER:?missing adapter}"; : "${CAL_GUIDANCE:?missing guidance}"; : "${CAL_GATE_START:?missing gate start}"; : "${CAL_SEED:?missing seed}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; ANALYSIS="$REPO/analysis/llm_feedback_20260922"; RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"; OUT="$RUN_ROOT/calibrated_gate_g${CAL_GUIDANCE}_t${CAL_GATE_START}_seed${CAL_SEED}_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/calibrated_active_set_probe.py" --repo "$REPO" \
 --config "$RUN_ROOT/full_train_1028011/config.pkl" --checkpoint "$RUN_ROOT/full_train_1028011/ema_model_epoch_8000.pt" \
 --adapter "$CAL_ADAPTER" --output "$OUT" --samples 1024 --steps 30 --guidance "$CAL_GUIDANCE" \
 --gate-start "$CAL_GATE_START" --seed "$CAL_SEED"
