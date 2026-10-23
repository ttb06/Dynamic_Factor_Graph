#!/usr/bin/env python3
"""Paired audit on rows whose active constraint topology changes mid-trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from typed_dynamic_graph_guidance import edge_masks


def row_violations(frame: pd.DataFrame, edges: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    applicable = np.zeros(len(frame), dtype=int)
    violations = np.zeros(len(frame), dtype=int)
    for edge in edges:
        if edge.get("usage_role", "actuate") != "actuate":
            continue
        active, _, bad = edge_masks(frame, edge)
        applicable += active.astype(int); violations += bad.astype(int)
    return applicable, violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--row-key", default="topology_changed_rows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    edges = json.loads(args.graph.read_text())
    trace = json.loads((args.dynamic / "trace_x0_reward_calibrated.json").read_text())
    changed = np.asarray(trace.get(args.row_key, []), dtype=int)
    paths = {
        "unconditional": args.dynamic / "samples/unconditional.csv",
        "dynamic": args.dynamic / "samples/x0_reward_calibrated.csv",
        "frozen": args.frozen / "samples/x0_reward_calibrated.csv",
        "final_repair": args.dynamic / "samples/final_reward_calibrated.csv",
    }
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    frozen_base = pd.read_csv(args.frozen / "samples/unconditional.csv")
    parity = bool(frames["unconditional"].astype(str).equals(frozen_base.astype(str)))
    if not parity:
        raise RuntimeError("dynamic/frozen unconditional samples are not exactly paired")
    base_app, base_bad = row_violations(frames["unconditional"], edges)
    rows = []
    groups = {"topology_changed": changed, "topology_stable": np.setdiff1d(np.arange(len(base_bad)), changed)}
    for group, indices in groups.items():
        for method, frame in frames.items():
            app, bad = row_violations(frame, edges)
            before = base_bad[indices]; after = bad[indices]
            rows.append({
                "group": group, "method": method, "rows": len(indices),
                "applicable_edge_events": int(app[indices].sum()),
                "violation_events": int(after.sum()),
                "violation_per_row": float(after.mean()) if len(indices) else np.nan,
                "rows_improved": int((after < before).sum()),
                "rows_worsened": int((after > before).sum()),
                "net_violation_events_removed": int((before - after).sum()),
            })
    result = pd.DataFrame(rows); result.to_csv(args.output / "flip_subgroup_metrics.csv", index=False)
    changed_result = result[result.group.eq("topology_changed")]
    lines = ["# Dynamic graph-state subgroup audit", "", f"Exact dynamic/frozen unconditional parity: `{parity}`. Row selector: `{args.row_key}`. Selected rows: `{len(changed)}`.", "",
             "| Method | Rows | Applicable events | Violation events | Violation/row | Improved | Worsened | Net removed |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in changed_result.itertuples(index=False):
        lines.append(f"| {row.method} | {row.rows} | {row.applicable_edge_events} | {row.violation_events} | {row.violation_per_row:.4f} | {row.rows_improved} | {row.rows_worsened} | {row.net_violation_events_removed} |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({"paired_parity": parity, "row_key": args.row_key, "selected_rows": len(changed)}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__": main()
