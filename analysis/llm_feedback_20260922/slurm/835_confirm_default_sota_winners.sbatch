#!/usr/bin/env bash
# Six-seed paired confirmation for the two promoted Default hybrid arms.
#SBATCH --job-name=llmtd-default-sota-confirm
#SBATCH --partition=prioritized
#SBATCH --array=0-11%12
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
GRAPH="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
CONFIG="$A/outputs/full_train_default_1033760/config.pkl"
CHECKPOINT="$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt"

VARIANTS=(joint_high pair300)
SEEDS=(20261371 20261372 20261373 20261374 20261375 20261376)
VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / 6))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 6))
VARIANT=${VARIANTS[$VARIANT_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
SAMPLES=8192
STEPS=50

NUM=.50
CAT=.25
MARG=.10
PAIR=2
TAIL=.10
case "$VARIANT" in
  joint_high) NUM=.65; CAT=.15; MARG=.20; PAIR=3; TAIL=.15 ;;
  pair300) PAIR=3 ;;
  *) echo "unknown variant" >&2; exit 64 ;;
esac

OUTROOT="$A/outputs/default_sota_confirm_${VARIANT}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT
cd "$REPO"

{
  printf 'stage=Default_hybrid_SOTA_six_seed_confirmation\n'
  printf 'job_id=%s\narray_job_id=%s\narray_task_id=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
  printf 'dataset=default\nvariant=%s\nseed=%s\nsamples=%s\nsteps=%s\n' "$VARIANT" "$SEED" "$SAMPLES" "$STEPS"
  printf 'numeric_strength=%s\ncategorical_strength=%s\nmarginal_scale=%s\npair_scale=%s\ntail=%s\n' "$NUM" "$CAT" "$MARG" "$PAIR" "$TAIL"
  printf 'screen_job=1110907\nselection_job=1110908\n'
  printf 'host=%s\nstarted_at=%s\n' "$(hostname)" "$(date --iso-8601=seconds)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'git_diff_sha256='; git diff --binary | sha256sum | cut -d' ' -f1
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu=/'
  sha256sum "$DRIVER" "$GRAPH" "$CONFIG" "$CHECKPOINT" "$A/outputs/default_sota_selection_1110907_1110908/selection.json"
} > "$PROVENANCE"

COMMON=(
  --repo "$REPO" --dataset default --graph "$GRAPH"
  --include-edge-ids old::r1_001,old::r1_014,old::r1_015,old::r1_017
  --config "$CONFIG" --checkpoint "$CHECKPOINT"
  --samples "$SAMPLES" --steps "$STEPS" --seed "$SEED"
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat
  --projection-strength 1 --projection-min-t .10 --projection-max-t .40
  --projection-interval-space edge_log_sigma_quantile --projection-schedule middle_peak
  --projection-route-mode auto_edge_valley_isotonic
  --projection-stability-mode rank --projection-stability-threshold .001
  --projection-persistence-steps 2 --projection-trust-radius .75
  --guidance-stage both --stochastic-sampler --second-order-correction
)

REFERENCE=(
  --distribution-transport-mode copula --distribution-transport-scope all
  --distribution-transport-strength .50 --distribution-transport-condition-target
  --distribution-transport-preserve-tail .10
  --distribution-transport-min-t .10 --distribution-transport-max-t .40
  --distribution-transport-schedule flat --distribution-transport-once
  --categorical-guidance-mode marginal_target --categorical-guidance-strength .25
  --categorical-guidance-min-t .20 --categorical-guidance-max-t .50
  --categorical-guidance-schedule middle_peak
  --categorical-guidance-marginal-scale .10 --categorical-guidance-pair-scale 2
)

CANDIDATE=(
  --distribution-transport-mode copula --distribution-transport-scope all
  --distribution-transport-strength "$NUM" --distribution-transport-condition-target
  --distribution-transport-preserve-tail "$TAIL"
  --distribution-transport-min-t .10 --distribution-transport-max-t .40
  --distribution-transport-schedule flat --distribution-transport-once
  --categorical-guidance-mode marginal_target --categorical-guidance-strength "$CAT"
  --categorical-guidance-min-t .20 --categorical-guidance-max-t .50
  --categorical-guidance-schedule middle_peak
  --categorical-guidance-marginal-scale "$MARG" --categorical-guidance-pair-scale "$PAIR"
)

REFERENCE_COMMAND=("$PY" -u "$DRIVER" "${COMMON[@]}" "${REFERENCE[@]}" --output "$OUTROOT/reference")
CANDIDATE_COMMAND=("$PY" -u "$DRIVER" "${COMMON[@]}" "${CANDIDATE[@]}" --output "$OUTROOT/candidate")
{
  printf 'reference_command='; printf '%q ' "${REFERENCE_COMMAND[@]}"; printf '\n'
  printf 'candidate_command='; printf '%q ' "${CANDIDATE_COMMAND[@]}"; printf '\n'
} > "$OUTROOT/command.txt"
"${REFERENCE_COMMAND[@]}"
"${CANDIDATE_COMMAND[@]}"

sha256sum "$OUTROOT/reference/samples/unconditional.csv" "$OUTROOT/candidate/samples/unconditional.csv" > "$OUTROOT/unconditional_sha256.txt"
if [[ "$(cut -d' ' -f1 "$OUTROOT/unconditional_sha256.txt" | sort -u | wc -l)" -ne 1 ]]; then
  echo "paired unconditional mismatch" >&2
  exit 65
fi

"$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$GRAPH" \
  --sample "B=$OUTROOT/reference/samples/unconditional.csv" \
  --sample "G=$OUTROOT/reference/samples/dense_factor.csv" \
  --sample "$VARIANT=$OUTROOT/candidate/samples/dense_factor.csv" \
  --seed "$SEED" --max-rows "$SAMPLES" --output "$OUTROOT/eval"
