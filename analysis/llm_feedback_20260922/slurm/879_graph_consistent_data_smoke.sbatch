#!/usr/bin/env bash
# Functional smoke on one existing pool; all data work stays on a Slurm node.
#SBATCH --job-name=llmtd-graph-data-smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SOURCE="$A/outputs/multidataset_particle_confirm_shoppers_proj_gt4_s20261461_1111943_0"
GRAPH="$A/outputs/shoppers_dynamic_confirm_s20261381_frozen_state_1067703_3/final_graph.json"
EDGE_IDS='r1_014,seed_003,r2_006,r1_015,r1_005,r1_021,r1_013'
OUT="$A/outputs/graph_consistent_data_smoke_${SLURM_JOB_ID}"

"$PY" -u "$A/scripts/graph_consistent_particle_resampling.py" \
  --repo "$REPO" --dataset shoppers \
  --pool "$SOURCE/pool/samples/dense_factor.csv" \
  --backbone-pool "$SOURCE/pool/samples/unconditional.csv" \
  --graph "$GRAPH" --include-edge-ids "$EDGE_IDS" --include-roles actuate \
  --count 4096 --bins 16 --pair-bins 4 --max-pairs 4 --target-pair-quota 2 \
  --pair-weight 1.5 --coverage-weight .30 --generic-rake-iterations 1 \
  --graph-rake-iterations 1 --graph-rake-damping .20 \
  --graph-relation-weight .5 --graph-coverage-weight 1 --graph-hard-weight 10 \
  --prior-index "$SOURCE/g_indices.csv" --prior-weight 1 \
  --prior-index "$SOURCE/tail20/selected_indices.csv" --prior-weight 2 \
  --consensus-strength 4 --temperature .5 --tail-mix .1 --draws 2 \
  --seed 20261461 --output "$OUT/selection"

"$PY" -u "$A/scripts/audit_graph_consistency.py" \
  --repo "$REPO" --dataset shoppers --pool "$SOURCE/pool/samples/dense_factor.csv" \
  --graph "$GRAPH" --include-edge-ids "$EDGE_IDS" --include-roles actuate --count 4096 \
  --sample-index "old=$SOURCE/winner/selected_indices.csv" \
  --sample-index "smoke=$OUT/selection/selected_indices.csv" --output "$OUT/audit"

test -s "$OUT/selection/selection_metadata.json"
test -s "$OUT/audit/graph_metrics.csv"
