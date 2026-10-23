#!/usr/bin/env bash
# Cross-dataset screen of the locked particle winners on baseline-rich datasets.
# Three paired base generators are tested per dataset; each task reuses one
# 8,192-row pool for five train-only 4,096-row selectors.
#SBATCH --job-name=llmtd-xdata-particle
#SBATCH --partition=prioritized
#SBATCH --array=0-8%9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
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
i=$SLURM_ARRAY_TASK_ID
dataset_index=$((i / 3))
base_index=$((i % 3))
BASES=(plain proj hybrid)
BASE=${BASES[$base_index]}

case "$dataset_index" in
  0)
    DATASET=shoppers; SEED=20261450
    GRAPH="$A/outputs/shoppers_dynamic_confirm_s20261381_frozen_state_1067703_3/final_graph.json"
    EVAL_GRAPH="$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json"
    CONFIG="$A/outputs/full_train_shoppers_1033761/config.pkl"
    CHECKPOINT="$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt"
    EDGE_IDS='r1_014,seed_003,r2_006,r1_015,r1_005,r1_021,r1_013'
    ;;
  1)
    DATASET=magic; SEED=20261451
    GRAPH="$A/outputs/extended_bscg_magic_1081009/graph.json"
    EVAL_GRAPH="$GRAPH"
    CONFIG="$A/outputs/extended_full_train_magic_1081010/config.pkl"
    CHECKPOINT="$A/outputs/extended_full_train_magic_1081010/ema_model_epoch_8000.pt"
    EDGE_IDS='bscg_000,bscg_001,bscg_002,bscg_003,bscg_004'
    ;;
  2)
    DATASET=diabetes; SEED=20261452
    GRAPH="$A/outputs/bootstrap_conditional_graph_diabetes_1070443_2/graph.json"
    EVAL_GRAPH="$GRAPH"
    CONFIG="$A/outputs/full_train_diabetes_1048479/config.pkl"
    CHECKPOINT="$A/outputs/full_train_diabetes_1048479/ema_model_epoch_8000.pt"
    EDGE_IDS='bscg_011'
    ;;
  *) echo "invalid dataset index" >&2; exit 64 ;;
esac

PROJECTION=(--projection-strength 0 --projection-min-t 0 --projection-max-t 0)
TRANSPORT=(--distribution-transport-mode none)
CATEGORICAL=(--categorical-guidance-mode none)
case "$BASE" in
  plain) ;;
  proj)
    PROJECTION=(--projection-strength 1 --projection-min-t .10 --projection-max-t .40)
    ;;
  hybrid)
    PROJECTION=(--projection-strength 1 --projection-min-t .10 --projection-max-t .40)
    TRANSPORT=(
      --distribution-transport-mode copula --distribution-transport-scope all
      --distribution-transport-condition-target --distribution-transport-preserve-tail .10
      --distribution-transport-strength .50 --distribution-transport-min-t .20
      --distribution-transport-max-t .35 --distribution-transport-schedule flat
      --distribution-transport-once
    )
    CATEGORICAL=(
      --categorical-guidance-mode marginal_target --categorical-guidance-strength .15
      --categorical-guidance-marginal-scale .10 --categorical-guidance-pair-scale 2
      --categorical-guidance-min-t .20 --categorical-guidance-max-t .50
      --categorical-guidance-schedule middle_peak
    )
    ;;
  *) echo "invalid base" >&2; exit 64 ;;
esac

OUTROOT="$A/outputs/multidataset_particle_${DATASET}_${BASE}_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT
cd "$REPO"

for path in "$DRIVER" "$RESAMPLER" "$GRAPH" "$EVAL_GRAPH" "$CONFIG" "$CHECKPOINT"; do
  [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 66; }
done
{
  printf 'stage=multidataset_particle_screen\n'
  printf 'job_id=%s\narray_job_id=%s\narray_task_id=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
  printf 'dataset=%s\nbase=%s\nseed=%s\npool_samples=8192\nselected_samples=4096\nsteps=50\n' "$DATASET" "$BASE" "$SEED"
  printf 'selectors=gt4,int20_t005,marg4,trend4,int50_t0\n'
  printf 'host=%s\nstarted_at=%s\n' "$(hostname)" "$(date --iso-8601=seconds)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'git_diff_sha256='; git diff --binary | sha256sum | cut -d' ' -f1
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu=/'
  sha256sum "$DRIVER" "$RESAMPLER" "$GRAPH" "$EVAL_GRAPH" "$CONFIG" "$CHECKPOINT"
} > "$PROVENANCE"

START=$(date +%s)
"$PY" -u "$DRIVER" --repo "$REPO" --dataset "$DATASET" \
  --graph "$GRAPH" --include-edge-ids "$EDGE_IDS" --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --samples 8192 --steps 50 --seed "$SEED" \
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .02 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  "${PROJECTION[@]}" --projection-interval-space edge_log_sigma_quantile \
  --projection-schedule middle_peak --projection-route-mode auto_edge_valley_isotonic \
  --projection-stability-mode rank --projection-stability-threshold .001 \
  --projection-persistence-steps 2 --projection-trust-radius .50 \
  "${TRANSPORT[@]}" "${CATEGORICAL[@]}" \
  --guidance-stage both --stochastic-sampler --second-order-correction \
  --output "$OUTROOT/pool"

POOL="$OUTROOT/pool/samples/dense_factor.csv"
BACKBONE="$OUTROOT/pool/samples/unconditional.csv"
"$PY" -u "$RESAMPLER" --repo "$REPO" --dataset "$DATASET" \
  --pool "$POOL" --backbone-pool "$BACKBONE" --count 4096 \
  --mode rake_marginal_pair_select --bins 16 --pair-bins 4 \
  --max-pairs 24 --target-pair-quota 8 --pair-weight 1.5 --coverage-weight .30 \
  --rake-iterations 5 --rake-damping .5 --weight-clip 20 \
  --temperature .35 --tail-mix .20 --draws 128 --seed "$SEED" --output "$OUTROOT/tail20"

"$PY" -u "$RESAMPLER" --repo "$REPO" --dataset "$DATASET" \
  --pool "$POOL" --backbone-pool "$BACKBONE" --count 4096 \
  --mode rake_marginal --bins 16 --pair-bins 4 --max-pairs 0 \
  --target-pair-quota 0 --pair-weight 0 --coverage-weight 0 \
  --rake-iterations 5 --rake-damping .5 --weight-clip 20 \
  --temperature .35 --tail-mix .20 --draws 64 --seed "$SEED" --output "$OUTROOT/margtail"

G_INDEX="$OUTROOT/g_indices.csv"
awk 'BEGIN { print "pool_index"; for (i=0; i<4096; i++) print i }' > "$G_INDEX"
COMMON=(--repo "$REPO" --dataset "$DATASET" --pool "$POOL" --backbone-pool "$BACKBONE"
  --count 4096 --mode consensus_rake --bins 16 --pair-bins 4
  --rake-iterations 5 --rake-damping .5 --weight-clip 20
  --temperature .50 --tail-mix .10 --draws 128 --consensus-strength 4
  --prior-index "$G_INDEX" --prior-weight 1 --seed "$SEED")
LOCKED=(--max-pairs 24 --target-pair-quota 8 --pair-weight 1.5 --coverage-weight .30
  --prior-index "$OUTROOT/tail20/selected_indices.csv" --prior-weight 2)

"$PY" -u "$RESAMPLER" "${COMMON[@]}" "${LOCKED[@]}" --output "$OUTROOT/gt4"
"$PY" -u "$RESAMPLER" "${COMMON[@]}" "${LOCKED[@]}" \
  --geometric-coverage-weight 20 --geometric-coverage-k 16 --geometric-coverage-tolerance 0 \
  --integrated-coverage-weight 20 --integrated-coverage-tolerance .005 \
  --integrated-coverage-shortlist 8 --integrated-coverage-folds 2 --output "$OUTROOT/int20_t005"
"$PY" -u "$RESAMPLER" "${COMMON[@]}" \
  --max-pairs 0 --target-pair-quota 0 --pair-weight 0 --coverage-weight 0 \
  --prior-index "$OUTROOT/margtail/selected_indices.csv" --prior-weight 2 --output "$OUTROOT/marg4"
"$PY" -u "$RESAMPLER" "${COMMON[@]}" \
  --max-pairs 32 --target-pair-quota 16 --pair-weight 3 --coverage-weight .15 \
  --prior-index "$OUTROOT/tail20/selected_indices.csv" --prior-weight 2 --output "$OUTROOT/trend4"
"$PY" -u "$RESAMPLER" "${COMMON[@]}" "${LOCKED[@]}" \
  --geometric-coverage-weight 20 --geometric-coverage-k 16 --geometric-coverage-tolerance 0 \
  --integrated-coverage-weight 50 --integrated-coverage-tolerance 0 \
  --integrated-coverage-shortlist 8 --integrated-coverage-folds 2 --output "$OUTROOT/int50_t0"

AFTER_SELECTION=$(date +%s)
printf 'pre_eval_runtime_seconds=%s\n' "$((AFTER_SELECTION - START))" >> "$PROVENANCE"
"$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset "$DATASET" --graph "$EVAL_GRAPH" --max-rows 4096 --seed "$SEED" \
  --sample "B=$OUTROOT/tail20/b_reference.csv" \
  --sample "G=$OUTROOT/tail20/g_reference.csv" \
  --sample "R=$OUTROOT/tail20/uniform_control.csv" \
  --sample "gt4=$OUTROOT/gt4/selected.csv" \
  --sample "int20_t005=$OUTROOT/int20_t005/selected.csv" \
  --sample "marg4=$OUTROOT/marg4/selected.csv" \
  --sample "trend4=$OUTROOT/trend4/selected.csv" \
  --sample "int50_t0=$OUTROOT/int50_t0/selected.csv" \
  --output "$OUTROOT/eval"
