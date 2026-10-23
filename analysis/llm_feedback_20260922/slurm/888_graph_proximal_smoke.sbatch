#!/usr/bin/env bash
#SBATCH --job-name=llmtd-graph-prox-smoke
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
bash -n "$A/slurm/886_graph_proximal_screen.sbatch"
bash -n "$A/slurm/887_summarize_graph_proximal_screen.sbatch"
"$PY" -m py_compile "$A/scripts/graph_proximal_swap_repair.py" "$A/scripts/summarize_graph_pareto_screen.py"
SOURCE="$A/outputs/multidataset_particle_confirm_shoppers_proj_gt4_s20261461_1111943_0"
OUT="$A/outputs/graph_proximal_smoke_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" -u "$A/scripts/graph_proximal_swap_repair.py" \
  --repo "$REPO" --dataset shoppers \
  --pool "$SOURCE/pool/samples/dense_factor.csv" \
  --backbone-pool "$SOURCE/pool/samples/unconditional.csv" \
  --graph "$A/outputs/shoppers_dynamic_confirm_s20261381_frozen_state_1067703_3/final_graph.json" \
  --anchor-index "$SOURCE/winner/selected_indices.csv" \
  --include-edge-ids r1_014,seed_003,r2_006,r1_015,r1_005,r1_021,r1_013 --include-roles actuate \
  --count 4096 --bins 16 --pair-bins 4 --max-pairs 24 --target-pair-quota 8 \
  --pair-weight 1.5 --coverage-weight .30 --required-graph-gain .001 \
  --swap-fraction .01 --quality-relative-tolerance .02 \
  --draws 8 --seed 20261461 --output "$OUT"
