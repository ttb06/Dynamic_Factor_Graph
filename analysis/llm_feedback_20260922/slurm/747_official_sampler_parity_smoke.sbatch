#!/usr/bin/env bash
# Verify that the custom guided loop reduces to the released TabDiff sampler
# when all graph action strengths are zero.
#SBATCH --job-name=llmtd-official-parity
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
OUT="$A/outputs/official_sampler_parity_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news \
  --graph "$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json" \
  --include-families llm_directional_bound --family-strengths llm_directional_bound=1 \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 128 --steps 20 --seed 20261609 --trees 0 --strength 0 \
  --factor-min-t 0 --factor-max-t 0 --projection-strength 0 \
  --stochastic-sampler --second-order-correction --output "$OUT"

"$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
out=Path(sys.argv[1])
a=pd.read_csv(out/'samples/unconditional.csv')
b=pd.read_csv(out/'samples/dense_factor.csv')
numeric=[]
for column in a:
    x=pd.to_numeric(a[column],errors='coerce'); y=pd.to_numeric(b[column],errors='coerce')
    if x.notna().all() and y.notna().all(): numeric.append(float(np.max(np.abs(x-y))))
same_strings=bool((a.astype(str).to_numpy()==b.astype(str).to_numpy()).all())
result={'rows':len(a),'max_numeric_abs_difference':max(numeric,default=0.0),
        'csv_string_exact':same_strings,'parity_pass':same_strings or max(numeric,default=0.0)<1e-6}
(out/'parity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if not result['parity_pass']: raise SystemExit(1)
PY
