#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cuts-de-split
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
cd "$ROOT/external/cuts"
"$ROOT/baseline_runs/envs/cuts/bin/python" split_datasets.py --dataset German --random_seed 42 --split_ratio 0.2
