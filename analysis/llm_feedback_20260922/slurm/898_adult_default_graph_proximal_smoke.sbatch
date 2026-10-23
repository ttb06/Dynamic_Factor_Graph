#!/usr/bin/env bash
# Syntax/input smoke for the Adult + Default graph-proximal transfer sweep.
#SBATCH --job-name=llmtd-addef-gp-smoke
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

cd "$REPO"
bash -n "$A/slurm/899_adult_graph_proximal_transfer_sweep.sbatch"
bash -n "$A/slurm/900_default_graph_proximal_transfer_sweep.sbatch"
bash -n "$A/slurm/901_summarize_adult_default_graph_proximal.sbatch"
"$PY" -m py_compile \
  "$A/scripts/dense_numeric_factor_messages_arr.py" \
  "$A/scripts/balanced_particle_resampling.py" \
  "$A/scripts/graph_proximal_swap_repair.py" \
  "$A/scripts/audit_graph_consistency.py" \
  "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  "$A/scripts/summarize_graph_pareto_screen.py"

inputs=(
  "$A/outputs/extended_bscg_adult_1081004/graph.json"
  "$A/outputs/extended_full_train_adult_1081005/config.pkl"
  "$A/outputs/extended_full_train_adult_1081005/ema_model_epoch_8000.pt"
  "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
)
for task in 0 1 2; do
  seed=$((20261421 + task))
  source="$A/outputs/default_intcov_confirm_s${seed}_1111817_${task}"
  inputs+=(
    "$source/pool/samples/dense_factor.csv"
    "$source/pool/samples/unconditional.csv"
    "$source/int20_t005/selected_indices.csv"
  )
done
for path in "${inputs[@]}"; do
  [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 66; }
done
printf 'smoke_ok=1\nhost=%s\nfinished_at=%s\n' "$(hostname)" "$(date --iso-8601=seconds)"
