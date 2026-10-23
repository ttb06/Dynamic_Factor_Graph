#!/usr/bin/env bash
# Syntax plus data smoke for deficit-activated graph-proximal repair.
#SBATCH --job-name=llmtd-gp-gate-smoke
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SOURCE="$A/outputs/graph_proximal_transfer_adult_s20261503_1114275_2"
OUT="$A/outputs/graph_deficit_gate_smoke_${SLURM_JOB_ID}"

bash -n "$A/slurm/903_confirm_graph_deficit_gate.sbatch"
bash -n "$A/slurm/904_summarize_graph_deficit_gate_confirmation.sbatch"
"$PY" -m py_compile "$A/scripts/graph_proximal_swap_repair.py" \
  "$A/scripts/summarize_graph_pareto_confirmation.py"
cd "$REPO"
"$PY" -u "$A/scripts/graph_proximal_swap_repair.py" \
  --repo "$REPO" --dataset adult \
  --pool "$SOURCE/pool/samples/dense_factor.csv" \
  --backbone-pool "$SOURCE/pool/samples/unconditional.csv" \
  --graph "$A/outputs/extended_bscg_adult_1081004/graph.json" \
  --anchor-index "$SOURCE/old/selected_indices.csv" \
  --include-edge-ids bscg_001,bscg_003 --include-roles actuate \
  --count 4096 --bins 16 --pair-bins 4 --max-pairs 24 --target-pair-quota 8 \
  --pair-weight 1.5 --coverage-weight .30 --selection-priority quality \
  --swap-fraction .0025 --quality-relative-tolerance .015 \
  --required-graph-gain .05 --graph-activation-threshold .001 \
  --draws 8 --seed 20261503 --output "$OUT/graph"
grep -q '"activation_skipped": true' "$OUT/graph/selection_metadata.json"
printf 'smoke_ok=1\nexpected_noop_gate=1\nhost=%s\nfinished_at=%s\n' \
  "$(hostname)" "$(date --iso-8601=seconds)" > "$OUT/RESULTS.txt"

