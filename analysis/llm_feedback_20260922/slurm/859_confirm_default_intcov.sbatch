#!/usr/bin/env bash
# Six fresh-pool confirmation of locked integrated-coverage arm int20_t005.
#SBATCH --job-name=llmtd-intcov-confirm
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
GRAPH="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
CONFIG="$A/outputs/full_train_default_1033760/config.pkl"
CHECKPOINT="$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt"
SEEDS=(20261421 20261422 20261423 20261424 20261425 20261426)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUTROOT="$A/outputs/default_intcov_confirm_s${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT
cd "$REPO"

{
  printf 'stage=Default_integrated_coverage_fresh_confirmation\n'
  printf 'job_id=%s\narray_job_id=%s\narray_task_id=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
  printf 'dataset=default\nvariant=int20_t005\nmode=consensus_rake\nseed=%s\n' "$SEED"
  printf 'pool_samples=16384\nselected_samples=8192\nsteps=50\ndraws=128\n'
  printf 'consensus_strength=4\npriors=G:1,tail20:2\n'
  printf 'geometric_coverage_weight=20\ngeometric_coverage_tolerance=0\n'
  printf 'integrated_coverage_weight=20\nintegrated_coverage_tolerance=.005\nintegrated_coverage_shortlist=8\nintegrated_coverage_folds=2\n'
  printf 'design_source=1111770\nhost=%s\nstarted_at=%s\n' "$(hostname)" "$(date --iso-8601=seconds)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu=/'
  sha256sum "$DRIVER" "$RESAMPLER" "$GRAPH" "$CONFIG" "$CHECKPOINT" \
    "$A/outputs/default_intcov_selection_1111770_1111771/selection.json"
} > "$PROVENANCE"

START=$(date +%s)
"$PY" -u "$DRIVER" --repo "$REPO" --dataset default --graph "$GRAPH" \
  --include-edge-ids old::r1_001,old::r1_014,old::r1_015,old::r1_017 \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --samples 16384 --steps 50 --seed "$SEED" \
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --projection-strength 1 --projection-min-t .10 --projection-max-t .40 \
  --projection-interval-space edge_log_sigma_quantile --projection-schedule middle_peak \
  --projection-route-mode auto_edge_valley_isotonic \
  --projection-stability-mode rank --projection-stability-threshold .001 \
  --projection-persistence-steps 2 --projection-trust-radius .75 \
  --guidance-stage both --stochastic-sampler --second-order-correction \
  --distribution-transport-mode copula --distribution-transport-scope all \
  --distribution-transport-strength .50 --distribution-transport-condition-target \
  --distribution-transport-preserve-tail .10 \
  --distribution-transport-min-t .10 --distribution-transport-max-t .40 \
  --distribution-transport-schedule flat --distribution-transport-once \
  --categorical-guidance-mode marginal_target --categorical-guidance-strength .25 \
  --categorical-guidance-min-t .20 --categorical-guidance-max-t .50 \
  --categorical-guidance-schedule middle_peak \
  --categorical-guidance-marginal-scale .10 --categorical-guidance-pair-scale 2 \
  --output "$OUTROOT/pool"

"$PY" -u "$RESAMPLER" --repo "$REPO" --dataset default \
  --pool "$OUTROOT/pool/samples/dense_factor.csv" --backbone-pool "$OUTROOT/pool/samples/unconditional.csv" \
  --count 8192 --mode uniform_search --bins 16 --pair-bins 4 --max-pairs 0 \
  --target-pair-quota 0 --pair-weight .75 --coverage-weight 0 \
  --temperature .5 --tail-mix .1 --draws 1 --seed "$SEED" --output "$OUTROOT/control"

"$PY" -u "$RESAMPLER" --repo "$REPO" --dataset default \
  --pool "$OUTROOT/pool/samples/dense_factor.csv" --backbone-pool "$OUTROOT/pool/samples/unconditional.csv" \
  --count 8192 --mode rake_marginal_pair_select --bins 16 --pair-bins 4 \
  --max-pairs 24 --target-pair-quota 8 --pair-weight 1.5 --coverage-weight .30 \
  --rake-iterations 5 --rake-damping .5 --weight-clip 20 \
  --temperature .35 --tail-mix .20 --draws 128 --seed "$SEED" --output "$OUTROOT/tail20"

G_INDEX="$OUTROOT/g_indices.csv"
awk 'BEGIN { print "pool_index"; for (i=0; i<8192; i++) print i }' > "$G_INDEX"
"$PY" -u "$RESAMPLER" --repo "$REPO" --dataset default \
  --pool "$OUTROOT/pool/samples/dense_factor.csv" --backbone-pool "$OUTROOT/pool/samples/unconditional.csv" \
  --count 8192 --mode consensus_rake --bins 16 --pair-bins 4 \
  --max-pairs 24 --target-pair-quota 8 --pair-weight 1.5 --coverage-weight .30 \
  --rake-iterations 5 --rake-damping .5 --weight-clip 20 \
  --temperature .50 --tail-mix .10 --draws 128 --consensus-strength 4 \
  --prior-index "$G_INDEX" --prior-weight 1 \
  --prior-index "$OUTROOT/tail20/selected_indices.csv" --prior-weight 2 \
  --geometric-coverage-weight 20 --geometric-coverage-k 16 --geometric-coverage-tolerance 0 \
  --integrated-coverage-weight 20 --integrated-coverage-tolerance .005 \
  --integrated-coverage-shortlist 8 --integrated-coverage-folds 2 \
  --seed "$SEED" --output "$OUTROOT/int20_t005"

AFTER_SELECTION=$(date +%s)
printf 'pre_eval_runtime_seconds=%s\n' "$((AFTER_SELECTION - START))" >> "$PROVENANCE"
"$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$GRAPH" \
  --sample "B=$OUTROOT/control/b_reference.csv" \
  --sample "G=$OUTROOT/control/g_reference.csv" \
  --sample "R=$OUTROOT/control/selected.csv" \
  --sample "int20_t005=$OUTROOT/int20_t005/selected.csv" \
  --seed "$SEED" --output "$OUTROOT/eval"
