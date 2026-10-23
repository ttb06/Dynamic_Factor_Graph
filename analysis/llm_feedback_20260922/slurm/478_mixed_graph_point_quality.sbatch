#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mixgraph-pointq
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
pre="$A/outputs/mixed_graph_policy_pretrained_point_1067458_0"
cont="$A/outputs/mixed_graph_policy_full_continue_point_1067458_2"
graph="$A/outputs/mixed_graph_policy_full_calibrated_point_1067458_4"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$A/outputs/natural_dag_stable_v3_20261010/graph.json" \
  --sample "tabdiff=$pre/samples/unconditional.csv" \
  --sample "pretrained_point=$pre/samples/xstate_reward_calibrated.csv" \
  --sample "full_continue=$cont/samples/unconditional.csv" \
  --sample "full_continue_point=$cont/samples/xstate_reward_calibrated.csv" \
  --sample "graph_trained=$graph/samples/unconditional.csv" \
  --sample "graph_trained_point=$graph/samples/xstate_reward_calibrated.csv" \
  --sample "graph_trained_final=$graph/samples/final_reward_calibrated.csv" \
  --seed 20261376 --output "$A/outputs/mixed_graph_point_quality_${SLURM_JOB_ID}"
