#!/usr/bin/env bash
#SBATCH --job-name=llmtd-xdata-smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"

bash -n "$A/slurm/869_select_multidataset_particle.sbatch"
bash -n "$A/slurm/870_confirm_multidataset_particle.sbatch"
bash -n "$A/slurm/871_summarize_multidataset_particle.sbatch"
"$PY" -m py_compile "$A/scripts/multidataset_particle_pipeline.py"
sha256sum "$A/scripts/multidataset_particle_pipeline.py" \
  "$A/slurm/869_select_multidataset_particle.sbatch" \
  "$A/slurm/870_confirm_multidataset_particle.sbatch" \
  "$A/slurm/871_summarize_multidataset_particle.sbatch"
