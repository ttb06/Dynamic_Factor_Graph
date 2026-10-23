#!/usr/bin/env bash
#SBATCH --job-name=llmtd-decouple
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-11
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(shoppers shoppers shoppers shoppers shoppers shoppers default default default default default default)
kinds=(global global global conditional conditional conditional global global global conditional conditional conditional)
times=(.25 .25 .25 .25 .25 .25 .40 .40 .40 .40 .40 .40)
seeds=(20261444 20261445 20261446 20261444 20261445 20261446 20261444 20261445 20261446 20261444 20261445 20261446)
source_runs=(full_train_shoppers_1033761 full_train_shoppers_1033761 full_train_shoppers_1033761 full_train_shoppers_1033761 full_train_shoppers_1033761 full_train_shoppers_1033761 full_train_default_1033760 full_train_default_1033760 full_train_default_1033760 full_train_default_1033760 full_train_default_1033760 full_train_default_1033760)
graph_indices=(1 1 1 1 1 1 0 0 0 0 0 0)
i=$SLURM_ARRAY_TASK_ID; dataset=${datasets[$i]}; kind=${kinds[$i]}; checkpoint_time=${times[$i]}; seed=${seeds[$i]}
source_run=${source_runs[$i]}; graph_index=${graph_indices[$i]}
common=(--repo "$REPO" --dataset "$dataset" --trigger mid --proposal-source final
  --graph "$A/outputs/bootstrap_conditional_graph_${dataset}_1070443_${graph_index}/graph.json"
  --config "$A/outputs/$source_run/config.pkl" --checkpoint "$A/outputs/$source_run/ema_model_epoch_8000.pt"
  --checkpoint-time "$checkpoint_time" --max-calibration-rate .05 --branches 8 --rounds 2
  --density-weight 1.0 --samples 4096 --steps 50 --seed "$seed"
  --output "$A/outputs/decoupled_activation_${kind}_${dataset}_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}")
cd "$REPO"
if [[ "$kind" == global ]]; then
  "$PY" "$A/scripts/dynamic_graph_conditional_repaint.py" "${common[@]}"
else
  "$PY" "$A/scripts/conditional_density_graph_repaint.py" "${common[@]}" --conditional-weight 1.0
fi
