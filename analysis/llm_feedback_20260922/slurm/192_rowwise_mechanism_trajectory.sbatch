#!/usr/bin/env bash
#SBATCH --job-name=llmtd-rowmech-traj
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CURATE_JOB_ID:?CURATE_JOB_ID required}"
: "${SAMPLES:=2048}"; : "${PROBE_SEED:=20261006}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
cd "$REPO"
"$PY" "$A/scripts/typed_graph_trajectory_probe.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/rowwise_mechanism_graph_${CURATE_JOB_ID}/active_graph.json" \
  --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --samples "$SAMPLES" --steps 50 --seed "$PROBE_SEED" \
  --output "$A/outputs/rowwise_mechanism_trajectory_${SLURM_JOB_ID}"
