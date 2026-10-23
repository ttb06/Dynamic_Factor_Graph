#!/usr/bin/env bash
# CPU-only action-space consensus smoke.
#SBATCH --job-name=llmtd-consensus-smoke
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x-%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x-%j.err
set -euo pipefail
set +x

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SCRIPTS="$A/scripts"

"$PY" -m py_compile \
  "$SCRIPTS/consensus_guidance_transport.py" \
  "$SCRIPTS/dense_numeric_factor_messages_consensus.py" \
  "$SCRIPTS/select_consensus_candidates.py"

PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<'PY'
import numpy as np
import pandas as pd
from consensus_guidance_transport import ConsensusGuidanceTransport
from factorized_residual_witness import FactorizedResidualWitnessTransport

rng = np.random.default_rng(20261360)
n = 500
target = np.where(rng.random(n) > 0.5, "yes", "no")
x1 = 1.5 * (target == "yes") + rng.normal(0.0, 0.35, n)
x2 = 0.8 * x1 + rng.normal(0.0, 0.20, n)
train = pd.DataFrame({"x1": x1, "x2": x2, "target": target})
preview = train.copy()
preview["x1"] = preview["x1"] * 1.20 + 0.25
preview["x2"] = preview["x2"] * 0.75 - 0.20

witness = FactorizedResidualWitnessTransport(
    train, ["x1", "x2"], [{"columns": ["x1", "x2"]}],
    mode="marginal_pair", target_column="target", trees=16, min_r2=0.01,
    max_pair_heads=2, top_factors_per_row=1, top_rows_per_factor=0.20,
    min_score=0.02, clip=2.0, seed=20261360,
)

class Reference:
    columns = ["x1", "x2"]
    def apply(self, frame, strength):
        output = frame.copy()
        current = output[self.columns].to_numpy(float)
        delta = np.column_stack((np.full(len(frame), 0.10), np.full(len(frame), -0.10))) * strength
        output.loc[:, self.columns] = current + delta
        return output, np.abs(delta) > 0, {"transport_cell_events": int((np.abs(delta) > 0).sum())}

for mode in ("aligned", "pcgrad"):
    operator = ConsensusGuidanceTransport(
        train, Reference(), witness, mode=mode, witness_ratio=0.30,
        target_column="target",
    )
    output, changed, diagnostics = operator.apply(preview, 0.10)
    assert changed.shape == (n, 2)
    assert output["target"].equals(preview["target"])
    assert diagnostics["consensus_target_followed_reference"] == 1
    assert diagnostics["consensus_extra_to_reference_norm_mean"] <= 0.300001
    assert diagnostics["consensus_witness_selected_cells"] > 0
print("CONSENSUS_SMOKE_OK")
PY

PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$SCRIPTS/dense_numeric_factor_messages_consensus.py" --help \
  > /dev/null
printf 'CONSENSUS_SMOKE_OK\n'
