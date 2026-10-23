#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-8edge-c
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
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
seeds=(20261376 20261377 20261378)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
graph_dir="$A/outputs/natural_dag_multi_entry_v3_20261010"

"$PY" "$A/scripts/harpoon_graph_proxy.py" --repo "$REPO" --dataset default \
  --graph "$graph_dir/graph.json" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/mixed_graph_finetune_rerun_full_continue_1067447_2/ema_model_epoch_150.pt" \
  --samples 4096 --steps 50 --seed "$seed" \
  --strengths 8 --policies hard --active-min-t 0 --active-max-t 1 --wilson-z 1 \
  --output "$A/outputs/harpoon_default_8edge_confirm_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
