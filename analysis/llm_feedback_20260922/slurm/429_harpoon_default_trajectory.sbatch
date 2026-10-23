#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-trajectory
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${GRAPH_JOB_ID:?GRAPH_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"; PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
requests=(range category both); request=${requests[$SLURM_ARRAY_TASK_ID]}
"$PY" "$A/scripts/typed_graph_trajectory_probe.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/harpoon_default_graphs_${GRAPH_JOB_ID}/${request}.json" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261382 --precision .90 --recall .80 --min-final-positive 8 \
  --output "$A/outputs/harpoon_default_trajectory_${request}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
