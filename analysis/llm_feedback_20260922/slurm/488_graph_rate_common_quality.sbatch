#!/usr/bin/env bash
#SBATCH --job-name=llmtd-graph-rate-quality
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
base="$A/outputs/mixed_graph_splitmedian_full_continue_s20261376_1067476_3"
graph_rate="$A/outputs/graph_rate_median_graph_1067504_0"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$A/outputs/natural_dag_stable_v3_20261010/graph.json" \
  --sample "continued_unguided=$base/samples/unconditional.csv" \
  --sample "split_median_local=$base/samples/xstate_reward_calibrated.csv" \
  --sample "split_median_final=$base/samples/final_reward_calibrated.csv" \
  --sample "split_median_graph_rate=$graph_rate/samples/xstate_reward_calibrated.csv" \
  --seed 20261376 --output "$A/outputs/graph_rate_common_quality_${SLURM_JOB_ID}"
