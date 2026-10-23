#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cert-low-support
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
"$PY" "$A/scripts/train_trajectory_certifier.py" \
  --input "$A/outputs/default_certifier_calibration_1065403/trajectory_certifier_samples.npz" \
  --output "$A/outputs/default_trajectory_certifier_low_support_${SLURM_JOB_ID}" \
  --min-class 12 --max-iter 160 --seed 20261010
