#!/usr/bin/env bash
#SBATCH --job-name=llmtd-graph-pareto-smoke
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
bash -n "$A/slurm/880_graph_pareto_anchor_screen.sbatch"
bash -n "$A/slurm/881_summarize_graph_pareto_screen.sbatch"
"$PY" -m py_compile "$A/scripts/graph_pareto_anchored_resampling.py" "$A/scripts/summarize_graph_pareto_screen.py"
SOURCE="$A/outputs/multidataset_particle_confirm_shoppers_proj_gt4_s20261461_1111943_0"
GRAPH="$A/outputs/shoppers_dynamic_confirm_s20261381_frozen_state_1067703_3/final_graph.json"
OUT="$A/outputs/graph_pareto_smoke_${SLURM_JOB_ID}"
"$PY" -u "$A/scripts/graph_pareto_anchored_resampling.py" \
  --repo "$REPO" --dataset shoppers --pool "$SOURCE/pool/samples/dense_factor.csv" \
  --backbone-pool "$SOURCE/pool/samples/unconditional.csv" --graph "$GRAPH" \
  --anchor-index "$SOURCE/winner/selected_indices.csv" \
  --include-edge-ids 'r1_014,seed_003,r2_006,r1_015,r1_005,r1_021,r1_013' \
  --count 4096 --max-pairs 4 --target-pair-quota 2 \
  --prior-index "$SOURCE/g_indices.csv" --prior-weight 1 \
  --prior-index "$SOURCE/tail20/selected_indices.csv" --prior-weight 2 \
  --required-graph-gain 0 --quality-relative-tolerance .05 \
  --generic-rake-iterations 1 --graph-rake-iterations 1 --draws 4 \
  --seed 20261461 --output "$OUT"
test -s "$OUT/selection_metadata.json"
