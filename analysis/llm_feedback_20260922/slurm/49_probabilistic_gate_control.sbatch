#!/usr/bin/env bash
#SBATCH --job-name=llmtd-probgate
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${PROB_ADAPTER:?missing adapter}"; : "${PROB_GUIDANCE:?missing guidance}"; : "${PROB_GATE_START:?missing gate start}"; : "${PROB_SEED:?missing seed}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; ANALYSIS="$REPO/analysis/llm_feedback_20260922"; RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"; OUT="$RUN_ROOT/prob_gate_g${PROB_GUIDANCE}_t${PROB_GATE_START}_seed${PROB_SEED}_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/probabilistic_gate_control_probe.py" --repo "$REPO" \
 --config "$RUN_ROOT/full_train_1028011/config.pkl" --checkpoint "$RUN_ROOT/full_train_1028011/ema_model_epoch_8000.pt" \
 --adapter "$PROB_ADAPTER" --output "$OUT" --samples 1024 --steps 30 --guidance "$PROB_GUIDANCE" \
 --gate-start "$PROB_GATE_START" --seed "$PROB_SEED"
