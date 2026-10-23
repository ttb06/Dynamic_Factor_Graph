#!/usr/bin/env bash
# CPU-node integration smoke for train-only balanced particle resampling.
#SBATCH --job-name=llmtd-particle-smoke
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
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
SCRIPT="$A/scripts/balanced_particle_resampling.py"
SOURCE="$A/outputs/consensus_default_aligned15_1110859_4/reference/samples"
OUT="$A/outputs/balanced_particle_smoke_${SLURM_JOB_ID}"
mkdir -p "$OUT"

"$PY" -u "$SCRIPT" --repo "$REPO" --dataset default \
  --pool "$SOURCE/dense_factor.csv" --backbone-pool "$SOURCE/unconditional.csv" \
  --count 1024 --mode hybrid_logit --max-pairs 8 --rake-iterations 2 \
  --draws 4 --seed 20261379 --output "$OUT/hybrid_logit"

"$PY" -u "$SCRIPT" --repo "$REPO" --dataset default \
  --pool "$SOURCE/dense_factor.csv" --backbone-pool "$SOURCE/unconditional.csv" \
  --count 1024 --mode hybrid_xgb --max-pairs 8 --rake-iterations 2 \
  --draws 4 --seed 20261379 --output "$OUT/hybrid_xgb"

"$PY" - <<'PY' "$OUT"
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
for name in ("hybrid_logit", "hybrid_xgb"):
    metadata = json.loads((root / name / "selection_metadata.json").read_text())
    assert metadata["selected_rows"] == 1024
    assert metadata["no_duplicate_selected_rows"] is True
    assert metadata["selection"]["best"]["objective"] <= metadata["selection"]["median_objective"] + 1e-12
    assert (root / name / "selected.csv").is_file()
print("BALANCED_PARTICLE_SMOKE_OK")
PY

{
  printf 'stage=balanced_particle_smoke\njob_id=%s\nhost=%s\nfinished_at=%s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$(date --iso-8601=seconds)"
  sha256sum "$SCRIPT" "$SOURCE/dense_factor.csv" "$SOURCE/unconditional.csv"
} > "$OUT/provenance.txt"
