#!/usr/bin/env bash
#SBATCH --job-name=llmtd-dyncdd-eval
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SOURCE_JOB_ID=${SOURCE_JOB_ID:?SOURCE_JOB_ID required}
S="$A/outputs/dynamic_cdd_graph_${SOURCE_JOB_ID}/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/natural_dag_stable_v3_20261010/graph.json" \
  --sample "tabdiff=$S/unconditional.csv" \
  --sample "static_cdd_full_hard=$S/static_full_hard.csv" \
  --sample "static_cdd_delayed_hard=$S/static_delayed_hard.csv" \
  --sample "dynamic_transient_hard=$S/dynamic_transient_hard.csv" \
  --sample "dynamic_persistent_hard=$S/dynamic_persistent_hard.csv" \
  --sample "dynamic_persistent_calibrated_min_kl=$S/dynamic_persistent_calibrated_min_kl.csv" \
  --sample "dynamic_persistent_calibrated_severity=$S/dynamic_persistent_calibrated_severity.csv" \
  --sample "calibrated_final_repair=$S/calibrated_final_repair.csv" \
  --output "$A/outputs/dynamic_cdd_graph_eval_${SLURM_JOB_ID}"
