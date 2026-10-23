#!/usr/bin/env bash
# Adult transfer: dynamic categorical graph injection, generic C3 anchor, then
# sparse graph-proximal output repair. Three fresh paired seeds.
#SBATCH --job-name=llmtd-adult-gp-xfer
#SBATCH --partition=prioritized
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
set +x
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
DRIVER="$A/scripts/dense_numeric_factor_messages_arr.py"
RESAMPLER="$A/scripts/balanced_particle_resampling.py"
SELECTOR="$A/scripts/graph_proximal_swap_repair.py"
AUDITOR="$A/scripts/audit_graph_consistency.py"
GRAPH="$A/outputs/extended_bscg_adult_1081004/graph.json"
CONFIG="$A/outputs/extended_full_train_adult_1081005/config.pkl"
CHECKPOINT="$A/outputs/extended_full_train_adult_1081005/ema_model_epoch_8000.pt"
EDGE_IDS='bscg_001,bscg_003'
SEEDS=(20261501 20261502 20261503)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUTROOT="$A/outputs/graph_proximal_transfer_adult_s${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT

cd "$REPO"
{
  printf 'stage=adult_graph_proximal_transfer_screen\n'
  printf 'job_id=%s\narray_job_id=%s\narray_task_id=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
  printf 'dataset=adult\nbase=cat025\nold_selector=gt4\nseed=%s\n' "$SEED"
  printf 'pool_samples=8192\nselected_samples=4096\nsteps=50\nedge_ids=%s\n' "$EDGE_IDS"
  printf 'host=%s\nstarted_at=%s\ngit_commit=%s\n' "$(hostname)" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu=/'
  sha256sum "$DRIVER" "$RESAMPLER" "$SELECTOR" "$AUDITOR" "$GRAPH" "$CONFIG" "$CHECKPOINT"
} > "$PROVENANCE"

"$PY" -u "$DRIVER" --repo "$REPO" --dataset adult --graph "$GRAPH" \
  --include-edge-ids "$EDGE_IDS" --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --samples 8192 --steps 50 --seed "$SEED" \
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .02 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --projection-strength 0 --projection-min-t 0 --projection-max-t 0 \
  --distribution-transport-mode none \
  --categorical-guidance-mode marginal_target --categorical-guidance-strength .25 \
  --categorical-guidance-marginal-scale .10 --categorical-guidance-pair-scale 2 \
  --categorical-guidance-min-t .20 --categorical-guidance-max-t .50 \
  --categorical-guidance-schedule middle_peak \
  --guidance-stage both --stochastic-sampler --second-order-correction \
  --output "$OUTROOT/pool"

POOL="$OUTROOT/pool/samples/dense_factor.csv"
BACKBONE="$OUTROOT/pool/samples/unconditional.csv"
"$PY" -u "$RESAMPLER" --repo "$REPO" --dataset adult \
  --pool "$POOL" --backbone-pool "$BACKBONE" --count 4096 \
  --mode rake_marginal_pair_select --bins 16 --pair-bins 4 \
  --max-pairs 24 --target-pair-quota 8 --pair-weight 1.5 --coverage-weight .30 \
  --rake-iterations 5 --rake-damping .5 --weight-clip 20 \
  --temperature .35 --tail-mix .20 --draws 128 --seed "$SEED" --output "$OUTROOT/tail20"

G_INDEX="$OUTROOT/g_indices.csv"
awk 'BEGIN { print "pool_index"; for (i=0; i<4096; i++) print i }' > "$G_INDEX"
"$PY" -u "$RESAMPLER" --repo "$REPO" --dataset adult \
  --pool "$POOL" --backbone-pool "$BACKBONE" --count 4096 \
  --mode consensus_rake --bins 16 --pair-bins 4 \
  --max-pairs 24 --target-pair-quota 8 --pair-weight 1.5 --coverage-weight .30 \
  --rake-iterations 5 --rake-damping .5 --weight-clip 20 \
  --temperature .50 --tail-mix .10 --draws 128 --consensus-strength 4 \
  --prior-index "$G_INDEX" --prior-weight 1 \
  --prior-index "$OUTROOT/tail20/selected_indices.csv" --prior-weight 2 \
  --seed "$SEED" --output "$OUTROOT/old"

COMMON=(--repo "$REPO" --dataset adult --pool "$POOL" --backbone-pool "$BACKBONE"
  --graph "$GRAPH" --anchor-index "$OUTROOT/old/selected_indices.csv"
  --include-edge-ids "$EDGE_IDS" --include-roles actuate --count 4096
  --bins 16 --pair-bins 4 --max-pairs 24 --target-pair-quota 8
  --pair-weight 1.5 --coverage-weight .30 --selection-priority quality
  --swap-temperature .75 --uniform-mix .10 --coverage-boost 2 --hard-boost 2
  --draws 256 --seed "$SEED")
run_variant() {
  local name=$1 swap=$2 quality=$3 gain=$4
  "$PY" -u "$SELECTOR" "${COMMON[@]}" \
    --swap-fraction "$swap" --quality-relative-tolerance "$quality" \
    --required-graph-gain "$gain" --output "$OUTROOT/$name"
}
run_variant pqs0025_g05 .0025 .015 .05
run_variant pqs005_g10 .005 .03 .10
run_variant pqs010_g10 .010 .06 .10
run_variant pqs020_g25 .020 .12 .25

AUDIT_ARGS=(--repo "$REPO" --dataset adult --pool "$POOL" --graph "$GRAPH"
  --include-edge-ids "$EDGE_IDS" --include-roles actuate --count 4096
  --sample-index "old=$OUTROOT/old/selected_indices.csv")
for method in pqs0025_g05 pqs005_g10 pqs010_g10 pqs020_g25; do
  AUDIT_ARGS+=(--sample-index "$method=$OUTROOT/$method/selected_indices.csv")
done
"$PY" -u "$AUDITOR" "${AUDIT_ARGS[@]}" --output "$OUTROOT/graph_audit"
"$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset adult --graph "$GRAPH" --max-rows 4096 --seed "$SEED" \
  --sample "B=$OUTROOT/tail20/b_reference.csv" \
  --sample "G=$OUTROOT/tail20/g_reference.csv" \
  --sample "R=$OUTROOT/tail20/uniform_control.csv" \
  --sample "old=$OUTROOT/old/selected.csv" \
  --sample "pqs0025_g05=$OUTROOT/pqs0025_g05/selected.csv" \
  --sample "pqs005_g10=$OUTROOT/pqs005_g10/selected.csv" \
  --sample "pqs010_g10=$OUTROOT/pqs010_g10/selected.csv" \
  --sample "pqs020_g25=$OUTROOT/pqs020_g25/selected.csv" --output "$OUTROOT/eval"
