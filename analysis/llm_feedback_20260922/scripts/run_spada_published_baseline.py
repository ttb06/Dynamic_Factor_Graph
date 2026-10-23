#!/usr/bin/env python3
"""Run SPADA with the Adult graph released in the authors' demonstration."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# Same semantic graph as response_income in the official SPADA notebook, with
# UCI/TabDiff column spelling adapted mechanically to this repository's CSV.
ADULT_RESPONSE = """
age: []
fnlwgt: []
education.num: [education->education.num, age->education.num]
capital.gain: [occupation->capital.gain, workclass->capital.gain, education->capital.gain, income->capital.gain]
capital.loss: [occupation->capital.loss, workclass->capital.loss, education->capital.loss, income->capital.loss]
hours.per.week: [occupation->hours.per.week, workclass->hours.per.week]
workclass: [occupation->workclass, income->workclass]
education: [education.num->education]
marital.status: [age->marital.status, relationship->marital.status]
occupation: [education->occupation, age->occupation, workclass->occupation]
relationship: [marital.status->relationship, sex->relationship]
race: []
sex: []
native.country: []
income: [education->income, workclass->income, occupation->income, capital.gain->income, capital.loss->income, hours.per.week->income]
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spada-repo", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.spada_repo.resolve()))
    import cvxpy as cp
    import graph
    from SPADA import SPADA

    # The paper uses the same binary cycle-removal ILP.  On hosts without a
    # commercial GUROBI license, solve that unchanged ILP with SciPy/HiGHS.
    installed = set(cp.installed_solvers())
    solver = "GUROBI" if "GUROBI" in installed else "SCIPY"
    graph.cp.GUROBI = solver

    started = time.time()
    model = SPADA(patient=args.patience, epoch=args.epochs)
    model.fit(
        data=str(args.train), description="Adult Income census dataset.",
        response=ADULT_RESPONSE, Tree=True,
    )
    train_seconds = time.time() - started
    sample_start = time.time()
    rows = model.sample(args.samples, method="nf")
    sample_seconds = time.time() - sample_start
    synthetic = pd.DataFrame(rows)
    reference_columns = pd.read_csv(args.train, nrows=0).columns.tolist()
    synthetic = synthetic[reference_columns]
    synthetic.to_csv(args.output / "spada.csv", index=False)
    summary = {
        "train_seconds": train_seconds, "sample_seconds": sample_seconds,
        "rows": len(synthetic), "cycle_ilp_solver": solver,
        "graph_source": "official SPADA demonstration response_income; column names mechanically adapted",
        "edges_after_cycle_removal": int(sum(len(v) for v in model.dependencies.values() if v and v != [""])),
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "llm_graph_response.txt").write_text(ADULT_RESPONSE + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
