#!/usr/bin/env bash
#SBATCH --job-name=llmtd-nollm-split
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
seeds=(20261006 20261007)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
BANK="$A/outputs/no_llm_bank_split_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
GRAPH="$A/outputs/no_llm_graph_split_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/mine_empirical_bank_no_llm.py" --repo "$REPO" --dataset default \
  --output "$BANK" --seed "$seed" --min-support 30 --maximum 30
"$PY" "$A/scripts/select_empirical_graph_no_llm.py" --source "$BANK/empirical_bank.json" \
  --info "$REPO/data/default/info.json" --output "$GRAPH" --max-edges 8 \
  --min-support 300 --min-lift 1.5 --max-rate .25 --max-spread .08
