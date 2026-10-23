#!/usr/bin/env bash
#SBATCH --job-name=llmtd-e2e-ckpt-traj
#SBATCH --partition=prioritized
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
EPOCHS=(100 200 300 400 500)
i=$SLURM_ARRAY_TASK_ID; epoch=${EPOCHS[$i]}
train="$A/outputs/e2e_graph_finetune_graph5_mid_1062230_2"
"$PY" "$A/scripts/typed_graph_trajectory_probe.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_catalog_v2_shoppers_1060953_0/constraint_catalog.json" \
  --config "$train/config.pkl" --checkpoint "$train/ema_model_epoch_${epoch}.pt" \
  --samples 8192 --steps 50 --seed 20261231 \
  --output "$A/outputs/e2e_checkpoint_trajectory_e${epoch}_${SLURM_ARRAY_JOB_ID}_${i}"
