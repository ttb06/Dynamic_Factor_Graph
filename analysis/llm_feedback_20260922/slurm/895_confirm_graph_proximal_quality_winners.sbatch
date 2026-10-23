#!/usr/bin/env bash
# Fresh-seed confirmation of quality-first proximal graph repair.
# Reuses already generated pools from job 876; no diffusion generation.
#SBATCH --job-name=llmtd-graph-pq-conf
#SBATCH --partition=prioritized
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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
: "${SOURCE_CONFIRM_JOB:?submit with SOURCE_CONFIRM_JOB}"
[[ "$SCREEN_JOB" =~ ^[0-9]+$ && "$SELECT_JOB" =~ ^[0-9]+$ && "$SOURCE_CONFIRM_JOB" =~ ^[0-9]+$ ]] || {
  echo "invalid job id" >&2; exit 64;
}
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SELECTOR="$A/scripts/graph_proximal_swap_repair.py"
AUDITOR="$A/scripts/audit_graph_consistency.py"
WINNERS="$A/outputs/graph_proximal_quality_selection_${SCREEN_JOB}_${SELECT_JOB}/winners.tsv"
[[ -s "$WINNERS" ]] || { echo "missing winners: $WINNERS" >&2; exit 66; }

i=$SLURM_ARRAY_TASK_ID
dataset_index=$((i / 3)); seed_index=$((i % 3))
case "$dataset_index" in
  0)
    DATASET=shoppers; SEEDS=(20261481 20261482 20261483)
    GRAPH="$A/outputs/shoppers_dynamic_confirm_s20261381_frozen_state_1067703_3/final_graph.json"
    EVAL_GRAPH="$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json"
    EDGE_IDS='r1_014,seed_003,r2_006,r1_015,r1_005,r1_021,r1_013'
    ;;
  1)
    DATASET=magic; SEEDS=(20261484 20261485 20261486)
    GRAPH="$A/outputs/extended_bscg_magic_1081009/graph.json"; EVAL_GRAPH="$GRAPH"
    EDGE_IDS='bscg_000,bscg_001,bscg_002,bscg_003,bscg_004'
    ;;
  *) echo "confirmation currently locks Shoppers and Magic only" >&2; exit 64 ;;
esac
SEED=${SEEDS[$seed_index]}
BASE=$(awk -F '\t' -v dataset="$DATASET" 'NR > 1 && $1 == dataset { print $2; exit }' "$WINNERS")
OLD_SELECTOR=$(awk -F '\t' -v dataset="$DATASET" 'NR > 1 && $1 == dataset { print $3; exit }' "$WINNERS")
METHOD=$(awk -F '\t' -v dataset="$DATASET" 'NR > 1 && $1 == dataset { print $4; exit }' "$WINNERS")
[[ -n "$BASE" && -n "$OLD_SELECTOR" && -n "$METHOD" ]] || { echo "winner missing for $DATASET" >&2; exit 66; }
case "$METHOD" in
  pqs005_g10) SWAP=.005; QUALITY=.03; GAIN=.10 ;;
  pqs005_g25) SWAP=.005; QUALITY=.03; GAIN=.25 ;;
  pqs010_g10) SWAP=.010; QUALITY=.06; GAIN=.10 ;;
  pqs010_g25) SWAP=.010; QUALITY=.06; GAIN=.25 ;;
  *) echo "unsupported quality-first proximal method: $METHOD" >&2; exit 64 ;;
esac

shopt -s nullglob
sources=("$A/outputs/graph_consistent_confirm_${DATASET}_"*_s"${SEED}_${SOURCE_CONFIRM_JOB}_${i}")
(( ${#sources[@]} == 1 )) || {
  printf 'expected one source for %s seed %s, got %s\n' "$DATASET" "$SEED" "${#sources[@]}" >&2
  printf '%s\n' "${sources[@]}" >&2
  exit 66
}
SOURCE=${sources[0]}
POOL="$SOURCE/pool/samples/dense_factor.csv"; BACKBONE="$SOURCE/pool/samples/unconditional.csv"
ANCHOR="$SOURCE/old/selected_indices.csv"
for path in "$POOL" "$BACKBONE" "$ANCHOR" "$GRAPH" "$EVAL_GRAPH"; do
  [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 66; }
done

OUTROOT="$A/outputs/graph_proximal_quality_confirm_${DATASET}_${METHOD}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT
cd "$REPO"
{
  printf 'stage=graph_proximal_quality_fresh_confirmation\n'
  printf 'screen_job=%s\nselect_job=%s\nsource_confirm_job=%s\n' "$SCREEN_JOB" "$SELECT_JOB" "$SOURCE_CONFIRM_JOB"
  printf 'job_id=%s\narray_job_id=%s\narray_task_id=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$i"
  printf 'dataset=%s\nbase=%s\nold_selector=%s\nmethod=%s\nseed=%s\nsource=%s\n' \
    "$DATASET" "$BASE" "$OLD_SELECTOR" "$METHOD" "$SEED" "$SOURCE"
  printf 'swap_fraction=%s\nquality_relative_tolerance=%s\nrequired_graph_gain=%s\n' "$SWAP" "$QUALITY" "$GAIN"
  printf 'host=%s\nstarted_at=%s\ngit_commit=%s\n' "$(hostname)" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu=/'
  sha256sum "$SELECTOR" "$AUDITOR" "$GRAPH" "$ANCHOR" "$POOL" "$WINNERS"
} > "$PROVENANCE"

"$PY" -u "$SELECTOR" --repo "$REPO" --dataset "$DATASET" --pool "$POOL" \
  --backbone-pool "$BACKBONE" --graph "$GRAPH" --anchor-index "$ANCHOR" \
  --include-edge-ids "$EDGE_IDS" --include-roles actuate --count 4096 \
  --bins 16 --pair-bins 4 --max-pairs 24 --target-pair-quota 8 \
  --pair-weight 1.5 --coverage-weight .30 --selection-priority quality \
  --swap-temperature .75 --uniform-mix .10 --coverage-boost 2 --hard-boost 2 \
  --swap-fraction "$SWAP" --quality-relative-tolerance "$QUALITY" \
  --required-graph-gain "$GAIN" --draws 256 --seed "$SEED" --output "$OUTROOT/graph"

"$PY" -u "$AUDITOR" --repo "$REPO" --dataset "$DATASET" --pool "$POOL" --graph "$GRAPH" \
  --include-edge-ids "$EDGE_IDS" --include-roles actuate --count 4096 \
  --sample-index "old=$ANCHOR" --sample-index "graph=$OUTROOT/graph/selected_indices.csv" \
  --output "$OUTROOT/graph_audit"
"$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset "$DATASET" --graph "$EVAL_GRAPH" --max-rows 4096 --seed "$SEED" \
  --sample "B=$SOURCE/tail20/b_reference.csv" --sample "G=$SOURCE/tail20/g_reference.csv" \
  --sample "R=$SOURCE/tail20/uniform_control.csv" --sample "old=$SOURCE/old/selected.csv" \
  --sample "graph=$OUTROOT/graph/selected.csv" --output "$OUTROOT/eval"
