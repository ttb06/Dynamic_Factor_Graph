#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mixgraph-traj
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
VARIANTS=(full_continue full_calibrated)
TRAIN_INDICES=(2 3)
i=$SLURM_ARRAY_TASK_ID
variant=${VARIANTS[$i]}
train_index=${TRAIN_INDICES[$i]}
checkpoint="$A/outputs/mixed_graph_finetune_rerun_${variant}_1067447_${train_index}/ema_model_epoch_150.pt"

"$PY" "$A/scripts/typed_graph_trajectory_probe.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/natural_dag_stable_v3_20261010/graph.json" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" --checkpoint "$checkpoint" \
  --samples 8192 --steps 50 --seed 20261392 --precision .90 --recall .80 --min-final-positive 8 \
  --output "$A/outputs/mixed_graph_trajectory_${variant}_${SLURM_ARRAY_JOB_ID}_${i}"
