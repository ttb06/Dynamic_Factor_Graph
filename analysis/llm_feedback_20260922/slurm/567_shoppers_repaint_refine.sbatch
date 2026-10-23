#!/usr/bin/env bash
#SBATCH --job-name=llmtd-repaint-refine
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-7
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
labels=(b8 b8_density1 b8_edit1 t20 t30 graph40 rounds3 margin05)
times=(.25 .25 .25 .20 .30 .25 .25 .25)
branches=(8 8 8 4 4 4 4 4)
density=(.5 1.0 .5 .5 .5 .5 .5 .5)
edit=(.5 .5 1.0 .5 .5 .5 .5 .5)
graph=(20 20 20 20 20 40 20 20)
rounds=(2 2 2 2 2 2 3 2)
margins=(0 0 0 0 0 0 0 .5)
i=$SLURM_ARRAY_TASK_ID; label=${labels[$i]}
cd "$REPO"
"$PY" "$A/scripts/dynamic_graph_conditional_repaint.py" \
  --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/bootstrap_conditional_graph_shoppers_1070443_1/graph.json" \
  --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --trigger mid --checkpoint-time "${times[$i]}" --max-calibration-rate .05 \
  --branches "${branches[$i]}" --rounds "${rounds[$i]}" \
  --graph-weight "${graph[$i]}" --density-weight "${density[$i]}" --edit-weight "${edit[$i]}" \
  --accept-margin "${margins[$i]}" --samples 2048 --steps 50 --seed 20261440 \
  --output "$A/outputs/dg_repaint_refine_shoppers_${label}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
