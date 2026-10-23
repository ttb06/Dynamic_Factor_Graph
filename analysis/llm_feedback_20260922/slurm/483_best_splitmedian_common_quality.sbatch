#!/usr/bin/env bash
#SBATCH --job-name=llmtd-best-common
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
pre="$A/outputs/mixed_graph_splitmedian_pretrained_s20261376_1067476_0"
cont="$A/outputs/mixed_graph_splitmedian_full_continue_s20261376_1067476_3"
robust="$A/outputs/mixed_graph_robust_confirm_s20261376_1067485_0"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$A/outputs/natural_dag_stable_v3_20261010/graph.json" \
  --sample "tabdiff=$pre/samples/unconditional.csv" \
  --sample "pretrained_splitmedian=$pre/samples/xstate_reward_calibrated.csv" \
  --sample "continued_splitmedian=$cont/samples/xstate_reward_calibrated.csv" \
  --sample "continued_final=$cont/samples/final_reward_calibrated.csv" \
  --sample "robust_graph_dynamic=$robust/samples/xstate_reward_calibrated.csv" \
  --sample "robust_graph_hybrid=$robust/samples/xstate_reward_calibrated_then_final_reward.csv" \
  --sample "robust_graph_final=$robust/samples/final_reward_calibrated.csv" \
  --seed 20261376 --output "$A/outputs/best_splitmedian_common_quality_${SLURM_JOB_ID}"
