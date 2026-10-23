#!/usr/bin/env bash
#SBATCH --job-name=llmtd-prior-proxy
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-3
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python

datasets=(shoppers shoppers diabetes diabetes)
variants=(continue semantic_prior continue semantic_prior)
runs=(full_train_shoppers_1033761 full_train_shoppers_1033761 full_train_diabetes_1048479 full_train_diabetes_1048479)
graphs=(
  "$A/outputs/shoppers_promote_confirm_s20261383_dynamic_1067743_0/evaluation_graph.json"
  "$A/outputs/shoppers_promote_confirm_s20261383_dynamic_1067743_0/evaluation_graph.json"
  "$A/outputs/diabetes_expanded_confirm_s20261383_dynamic_1068457_0/evaluation_graph.json"
  "$A/outputs/diabetes_expanded_confirm_s20261383_dynamic_1068457_0/evaluation_graph.json"
)
i=$SLURM_ARRAY_TASK_ID
dataset=${datasets[$i]}
variant=${variants[$i]}
run=${runs[$i]}
graph=${graphs[$i]}
out="$A/outputs/semantic_prior_proxy_${dataset}_${variant}_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$out"

"$PY" "$A/scripts/select_exact_verified_graph.py" \
  --repo "$REPO" --dataset "$dataset" --graph "$graph" --min-support 30 \
  --include-score-only --output "$out/exact_graph.json"

if [[ "$variant" == semantic_prior ]]; then
  app=.05; rel=.10; rate=.50; cons=.05
else
  app=0; rel=0; rate=0; cons=0
fi

"$PY" "$A/scripts/finetune_mixed_graph_calibrated_tabdiff.py" \
  --repo "$REPO" --dataset "$dataset" \
  --config "$A/outputs/$run/config.pkl" \
  --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
  --graph "$out/exact_graph.json" --epochs 150 --batch-size 4096 \
  --learning-rate 1e-6 --trainable-scope full \
  --applicability-weight "$app" --relation-weight "$rel" --rate-weight "$rate" \
  --consistency-weight "$cons" --calibration-target point \
  --t-min .10 --t-max .50 --truth-temperature .20 --save-every 150 --seed 20261020 \
  --output "$out/train"

mkdir -p "$out/samples"
"$PY" "$A/scripts/sample_tabdiff_checkpoint.py" \
  --repo "$REPO" --dataset "$dataset" --config "$A/outputs/$run/config.pkl" \
  --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261020 --output "$out/samples/pretrained.csv"
"$PY" "$A/scripts/sample_tabdiff_checkpoint.py" \
  --repo "$REPO" --dataset "$dataset" --config "$out/train/config.pkl" \
  --checkpoint "$out/train/ema_model_epoch_150.pt" \
  --samples 4096 --steps 50 --seed 20261020 --output "$out/samples/finetuned.csv"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset "$dataset" --graph "$graph" --seed 20261020 \
  --sample "pretrained=$out/samples/pretrained.csv" \
  --sample "$variant=$out/samples/finetuned.csv" \
  --output "$out/evaluation"
