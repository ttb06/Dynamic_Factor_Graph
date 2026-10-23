#!/bin/bash
#SBATCH --job-name=llmtd-stage-value
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
R=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff; A="$R/analysis/llm_feedback_20260922"; cd "$R"
/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python "$A/scripts/intervention_stage_value_audit.py" --default "$A/outputs/dynamic_graph_solver_default_1051370_0" --shoppers "$A/outputs/dynamic_graph_solver_shoppers_1051370_1" --output "$A/outputs/intervention_stage_value_${SLURM_JOB_ID}"
