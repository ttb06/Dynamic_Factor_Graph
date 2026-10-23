#!/usr/bin/env bash
#SBATCH --job-name=llmtd-e2e-common
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
pre="$A/outputs/e2e_graph_eval_pretrained_1062232_0/samples"
g5="$A/outputs/e2e_graph_eval_graph5_mid_1062232_3/samples"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" --seed 20261302 \
  --sample "TabDiff=$pre/unconditional.csv" \
  --sample "TabDiff+Dynamic=$pre/x0_reward_calibrated.csv" \
  --sample "TabDiff+FinalRepair=$pre/final_reward_calibrated.csv" \
  --sample "GraphFT=$g5/unconditional.csv" \
  --sample "GraphFT+Dynamic=$g5/x0_reward_calibrated.csv" \
  --sample "GraphFT+FinalRepair=$g5/final_reward_calibrated.csv" \
  --output "$A/outputs/e2e_common_metrics_${SLURM_JOB_ID}"
