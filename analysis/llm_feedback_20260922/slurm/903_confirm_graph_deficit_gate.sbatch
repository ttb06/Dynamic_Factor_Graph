#!/usr/bin/env bash
# Fresh/held-out confirmation of graph-deficit-activated pqs0025_g05.
# Adult uses three new pools; Default uses the three untouched pools from the
# six-seed int20_t005 confirmation.
#SBATCH --job-name=llmtd-gp-gate-conf
#SBATCH --partition=prioritized
#SBATCH --array=0-5%6
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
i=$SLURM_ARRAY_TASK_ID

if (( i < 3 )); then
  DATASET=adult; BASE=cat025; OLD_SELECTOR=gt4; COUNT=4096
  SEEDS=(20261511 20261512 20261513); SEED=${SEEDS[$i]}
  GRAPH="$A/outputs/extended_bscg_adult_1081004/graph.json"
  EVAL_GRAPH="$GRAPH"; EDGE_IDS='bscg_001,bscg_003'
  CONFIG="$A/outputs/extended_full_train_adult_1081005/config.pkl"
  CHECKPOINT="$A/outputs/extended_full_train_adult_1081005/ema_model_epoch_8000.pt"
  OUTROOT="$A/outputs/graph_deficit_gate_confirm_adult_s${SEED}_${SLURM_ARRAY_JOB_ID}_${i}"
  mkdir -p "$OUTROOT"
  PROVENANCE="$OUTROOT/provenance.txt"
  printf 'stage=graph_deficit_gate_confirmation\ndataset=adult\nseed=%s\nstarted_at=%s\n' \
    "$SEED" "$(date --iso-8601=seconds)" > "$PROVENANCE"
  trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT
  cd "$REPO"
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
  ANCHOR="$OUTROOT/old/selected_indices.csv"
  B_REF="$OUTROOT/tail20/b_reference.csv"; G_REF="$OUTROOT/tail20/g_reference.csv"
  R_REF="$OUTROOT/tail20/uniform_control.csv"; OLD_SAMPLE="$OUTROOT/old/selected.csv"
  INTEGRATED=()
  SOURCE=generated_fresh
else
  DATASET=default; BASE=hybrid; OLD_SELECTOR=int20_t005; COUNT=8192
  seed_index=$((i - 3)); SEEDS=(20261424 20261425 20261426); SEED=${SEEDS[$seed_index]}
  source_task=$((seed_index + 3))
  GRAPH="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
  EVAL_GRAPH="$GRAPH"; EDGE_IDS='old::r1_001,old::r1_014,old::r1_015,old::r1_017'
  SOURCE="$A/outputs/default_intcov_confirm_s${SEED}_1111817_${source_task}"
  OUTROOT="$A/outputs/graph_deficit_gate_confirm_default_s${SEED}_${SLURM_ARRAY_JOB_ID}_${i}"
  mkdir -p "$OUTROOT"
  PROVENANCE="$OUTROOT/provenance.txt"
  printf 'stage=graph_deficit_gate_confirmation\ndataset=default\nseed=%s\nstarted_at=%s\n' \
    "$SEED" "$(date --iso-8601=seconds)" > "$PROVENANCE"
  trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT
  POOL="$SOURCE/pool/samples/dense_factor.csv"
  BACKBONE="$SOURCE/pool/samples/unconditional.csv"
  ANCHOR="$SOURCE/int20_t005/selected_indices.csv"
  B_REF="$SOURCE/control/b_reference.csv"; G_REF="$SOURCE/control/g_reference.csv"
  R_REF="$SOURCE/control/uniform_control.csv"; OLD_SAMPLE="$SOURCE/int20_t005/selected.csv"
  INTEGRATED=(--geometric-coverage-weight 20 --geometric-coverage-k 16
    --geometric-coverage-tolerance 0 --integrated-coverage-weight 20
    --integrated-coverage-tolerance .005 --integrated-coverage-shortlist 8
    --integrated-coverage-folds 2)
  cd "$REPO"
fi

for path in "$POOL" "$BACKBONE" "$ANCHOR" "$GRAPH" "$B_REF" "$G_REF" "$R_REF" "$OLD_SAMPLE"; do
  [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 66; }
done
{
  printf 'stage=graph_deficit_gate_confirmation\n'
  printf 'job_id=%s\narray_job_id=%s\narray_task_id=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$i"
  printf 'dataset=%s\nbase=%s\nold_selector=%s\nmethod=pqs0025_g05_gate1e3\nseed=%s\nsource=%s\n' \
    "$DATASET" "$BASE" "$OLD_SELECTOR" "$SEED" "$SOURCE"
  printf 'graph_activation_threshold=.001\nswap_fraction=.0025\nquality_relative_tolerance=.015\nrequired_graph_gain=.05\n'
  printf 'host=%s\nstarted_at=%s\ngit_commit=%s\n' "$(hostname)" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu=/'
  sha256sum "$SELECTOR" "$AUDITOR" "$GRAPH" "$ANCHOR" "$POOL"
} > "$PROVENANCE"

"$PY" -u "$SELECTOR" --repo "$REPO" --dataset "$DATASET" --pool "$POOL" \
  --backbone-pool "$BACKBONE" --graph "$GRAPH" --anchor-index "$ANCHOR" \
  --include-edge-ids "$EDGE_IDS" --include-roles actuate --count "$COUNT" \
  --bins 16 --pair-bins 4 --max-pairs 24 --target-pair-quota 8 \
  --pair-weight 1.5 --coverage-weight .30 --selection-priority quality \
  --swap-temperature .75 --uniform-mix .10 --coverage-boost 2 --hard-boost 2 \
  --swap-fraction .0025 --quality-relative-tolerance .015 \
  --required-graph-gain .05 --graph-activation-threshold .001 \
  "${INTEGRATED[@]}" --draws 256 --seed "$SEED" --output "$OUTROOT/graph"

"$PY" -u "$AUDITOR" --repo "$REPO" --dataset "$DATASET" --pool "$POOL" --graph "$GRAPH" \
  --include-edge-ids "$EDGE_IDS" --include-roles actuate --count "$COUNT" \
  --sample-index "old=$ANCHOR" --sample-index "graph=$OUTROOT/graph/selected_indices.csv" \
  --output "$OUTROOT/graph_audit"
"$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset "$DATASET" --graph "$EVAL_GRAPH" --max-rows "$COUNT" --seed "$SEED" \
  --sample "B=$B_REF" --sample "G=$G_REF" --sample "R=$R_REF" \
  --sample "old=$OLD_SAMPLE" --sample "graph=$OUTROOT/graph/selected.csv" \
  --output "$OUTROOT/eval"

