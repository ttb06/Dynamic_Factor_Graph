#!/usr/bin/env bash
# Fresh-pool confirmation of the graph-consistent selector chosen in job 874.
#SBATCH --job-name=llmtd-graph-confirm
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

: "${SCREEN_JOB:?submit with SCREEN_JOB}"
: "${SELECT_JOB:?submit with SELECT_JOB}"
[[ "$SCREEN_JOB" =~ ^[0-9]+$ && "$SELECT_JOB" =~ ^[0-9]+$ ]] || { echo "invalid job id" >&2; exit 64; }
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
DRIVER="$A/scripts/dense_numeric_factor_messages_arr.py"
RESAMPLER="$A/scripts/balanced_particle_resampling.py"
SELECTOR="$A/scripts/graph_consistent_particle_resampling.py"
AUDITOR="$A/scripts/audit_graph_consistency.py"
WINNERS="$A/outputs/graph_consistent_selection_${SCREEN_JOB}_${SELECT_JOB}/winners.tsv"
[[ -s "$WINNERS" ]] || { echo "missing winners: $WINNERS" >&2; exit 66; }

i=$SLURM_ARRAY_TASK_ID
dataset_index=$((i / 3))
seed_index=$((i % 3))
case "$dataset_index" in
  0)
    DATASET=shoppers; SEEDS=(20261481 20261482 20261483)
    GRAPH="$A/outputs/shoppers_dynamic_confirm_s20261381_frozen_state_1067703_3/final_graph.json"
    EVAL_GRAPH="$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json"
    CONFIG="$A/outputs/full_train_shoppers_1033761/config.pkl"
    CHECKPOINT="$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt"
    EDGE_IDS='r1_014,seed_003,r2_006,r1_015,r1_005,r1_021,r1_013'
    ;;
  1)
    DATASET=magic; SEEDS=(20261484 20261485 20261486)
    GRAPH="$A/outputs/extended_bscg_magic_1081009/graph.json"
    EVAL_GRAPH="$GRAPH"
    CONFIG="$A/outputs/extended_full_train_magic_1081010/config.pkl"
    CHECKPOINT="$A/outputs/extended_full_train_magic_1081010/ema_model_epoch_8000.pt"
    EDGE_IDS='bscg_000,bscg_001,bscg_002,bscg_003,bscg_004'
    ;;
  2)
    DATASET=diabetes; SEEDS=(20261487 20261488 20261489)
    GRAPH="$A/outputs/bootstrap_conditional_graph_diabetes_1070443_2/graph.json"
    EVAL_GRAPH="$GRAPH"
    CONFIG="$A/outputs/full_train_diabetes_1048479/config.pkl"
    CHECKPOINT="$A/outputs/full_train_diabetes_1048479/ema_model_epoch_8000.pt"
    EDGE_IDS='bscg_011'
    ;;
  *) echo "invalid dataset index" >&2; exit 64 ;;
esac
SEED=${SEEDS[$seed_index]}
BASE=$(awk -F '\t' -v dataset="$DATASET" 'NR > 1 && $1 == dataset { print $2; exit }' "$WINNERS")
OLD_SELECTOR=$(awk -F '\t' -v dataset="$DATASET" 'NR > 1 && $1 == dataset { print $3; exit }' "$WINNERS")
METHOD=$(awk -F '\t' -v dataset="$DATASET" 'NR > 1 && $1 == dataset { print $4; exit }' "$WINNERS")
[[ -n "$BASE" && -n "$OLD_SELECTOR" && -n "$METHOD" ]] || { echo "winner missing for $DATASET" >&2; exit 66; }

PROJECTION=(--projection-strength 0 --projection-min-t 0 --projection-max-t 0)
TRANSPORT=(--distribution-transport-mode none)
CATEGORICAL=(--categorical-guidance-mode none)
case "$BASE" in
  proj) PROJECTION=(--projection-strength 1 --projection-min-t .10 --projection-max-t .40) ;;
  hybrid)
    PROJECTION=(--projection-strength 1 --projection-min-t .10 --projection-max-t .40)
    TRANSPORT=(--distribution-transport-mode copula --distribution-transport-scope all
      --distribution-transport-condition-target --distribution-transport-preserve-tail .10
      --distribution-transport-strength .50 --distribution-transport-min-t .20
      --distribution-transport-max-t .35 --distribution-transport-schedule flat
      --distribution-transport-once)
    CATEGORICAL=(--categorical-guidance-mode marginal_target --categorical-guidance-strength .15
      --categorical-guidance-marginal-scale .10 --categorical-guidance-pair-scale 2
      --categorical-guidance-min-t .20 --categorical-guidance-max-t .50
      --categorical-guidance-schedule middle_peak)
    ;;
  *) echo "unsupported base: $BASE" >&2; exit 64 ;;
esac

OUTROOT="$A/outputs/graph_consistent_confirm_${DATASET}_${METHOD}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT
cd "$REPO"
{
  printf 'stage=graph_consistent_fresh_confirmation\n'
  printf 'screen_job=%s\nselect_job=%s\njob_id=%s\narray_job_id=%s\narray_task_id=%s\n' \
    "$SCREEN_JOB" "$SELECT_JOB" "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
  printf 'dataset=%s\nbase=%s\nold_selector=%s\nmethod=%s\nseed=%s\n' \
    "$DATASET" "$BASE" "$OLD_SELECTOR" "$METHOD" "$SEED"
  printf 'pool_samples=8192\nselected_samples=4096\nsteps=50\ngraph=%s\nedge_ids=%s\n' "$GRAPH" "$EDGE_IDS"
  printf 'host=%s\nstarted_at=%s\ngit_commit=%s\n' "$(hostname)" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu=/'
  sha256sum "$DRIVER" "$RESAMPLER" "$SELECTOR" "$AUDITOR" "$GRAPH" "$EVAL_GRAPH" "$CONFIG" "$CHECKPOINT" "$WINNERS"
} > "$PROVENANCE"

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
G_INDEX="$OUTROOT/g_indices.csv"
awk 'BEGIN { print "pool_index"; for (i=0; i<4096; i++) print i }' > "$G_INDEX"

OLD_COMMON=(--repo "$REPO" --dataset "$DATASET" --pool "$POOL" --backbone-pool "$BACKBONE"
  --count 4096 --mode consensus_rake --bins 16 --pair-bins 4
  --rake-iterations 5 --rake-damping .5 --weight-clip 20
  --temperature .50 --tail-mix .10 --draws 128 --consensus-strength 4
  --prior-index "$G_INDEX" --prior-weight 1 --seed "$SEED"
  --max-pairs 24 --target-pair-quota 8 --pair-weight 1.5 --coverage-weight .30
  --prior-index "$OUTROOT/tail20/selected_indices.csv" --prior-weight 2)
OLD_EXTRA=()
if [[ "$OLD_SELECTOR" == int20_t005 ]]; then
  OLD_EXTRA=(--geometric-coverage-weight 20 --geometric-coverage-k 16 --geometric-coverage-tolerance 0
    --integrated-coverage-weight 20 --integrated-coverage-tolerance .005
    --integrated-coverage-shortlist 8 --integrated-coverage-folds 2)
elif [[ "$OLD_SELECTOR" != gt4 ]]; then
  echo "unsupported old selector: $OLD_SELECTOR" >&2; exit 64
fi
"$PY" -u "$RESAMPLER" "${OLD_COMMON[@]}" "${OLD_EXTRA[@]}" --output "$OUTROOT/old"

case "$METHOD" in
  gc_lite)   GRAPH_ARGS=(--graph-relation-weight .25 --graph-coverage-weight .50 --graph-hard-weight 5  --graph-rake-iterations 1 --graph-rake-damping .10) ;;
  gc_bal)    GRAPH_ARGS=(--graph-relation-weight .50 --graph-coverage-weight 1.0 --graph-hard-weight 10 --graph-rake-iterations 2 --graph-rake-damping .20) ;;
  gc_strong) GRAPH_ARGS=(--graph-relation-weight 1.0 --graph-coverage-weight 2.0 --graph-hard-weight 20 --graph-rake-iterations 3 --graph-rake-damping .25) ;;
  gc_cov)    GRAPH_ARGS=(--graph-relation-weight .50 --graph-coverage-weight 4.0 --graph-hard-weight 20 --graph-rake-iterations 2 --graph-rake-damping .20) ;;
  gc_hard)   GRAPH_ARGS=(--graph-relation-weight .50 --graph-coverage-weight 1.0 --graph-hard-weight 50 --graph-rake-iterations 2 --graph-rake-damping .20) ;;
  *) echo "unsupported graph method: $METHOD" >&2; exit 64 ;;
esac
GRAPH_COMMON=(--repo "$REPO" --dataset "$DATASET" --pool "$POOL" --backbone-pool "$BACKBONE"
  --graph "$GRAPH" --include-edge-ids "$EDGE_IDS" --include-roles actuate
  --count 4096 --bins 16 --pair-bins 4 --max-pairs 24 --target-pair-quota 8
  --pair-weight 1.5 --coverage-weight .30 --generic-weight 1
  --generic-rake-iterations 5 --generic-rake-damping .5 --weight-clip 20
  --temperature .50 --tail-mix .10 --draws 128 --consensus-strength 4
  --prior-index "$G_INDEX" --prior-weight 1
  --prior-index "$OUTROOT/tail20/selected_indices.csv" --prior-weight 2
  --seed "$SEED")
"$PY" -u "$SELECTOR" "${GRAPH_COMMON[@]}" "${OLD_EXTRA[@]}" "${GRAPH_ARGS[@]}" --output "$OUTROOT/graph"

"$PY" -u "$AUDITOR" --repo "$REPO" --dataset "$DATASET" --pool "$POOL" --graph "$GRAPH" \
  --include-edge-ids "$EDGE_IDS" --include-roles actuate --count 4096 \
  --sample-index "old=$OUTROOT/old/selected_indices.csv" \
  --sample-index "graph=$OUTROOT/graph/selected_indices.csv" --output "$OUTROOT/graph_audit"

"$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset "$DATASET" --graph "$EVAL_GRAPH" --max-rows 4096 --seed "$SEED" \
  --sample "B=$OUTROOT/tail20/b_reference.csv" \
  --sample "G=$OUTROOT/tail20/g_reference.csv" \
  --sample "R=$OUTROOT/tail20/uniform_control.csv" \
  --sample "old=$OUTROOT/old/selected.csv" \
  --sample "graph=$OUTROOT/graph/selected.csv" --output "$OUTROOT/eval"
