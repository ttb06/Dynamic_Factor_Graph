#!/usr/bin/env python3
"""Aggregate the general state-conditioned graph probes across datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ORDER = ["(0.75,1.0]", "(0.50,0.75]", "(0.25,0.50]", "(0.10,0.25]", "[0,0.10]"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--array-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    directories = sorted(args.root.glob(f"general_state_graph_*_{args.array_job}_*"))
    if not directories:
        raise RuntimeError("no completed state-graph outputs")
    row_frames, status_frames, selected_frames = [], [], []
    for directory in directories:
        metadata = json.loads((directory / "run_metadata.json").read_text())
        dataset = metadata["dataset"]
        row_frames.append(pd.read_csv(directory / "row_graph_summary.csv").assign(dataset=dataset))
        status_frames.append(pd.read_csv(directory / "state_graph_summary.csv").assign(dataset=dataset))
        selected_frames.append(pd.read_csv(directory / "selected_specification_graph.csv").assign(run_dataset=dataset))
    rows = pd.concat(row_frames, ignore_index=True)
    status = pd.concat(status_frames, ignore_index=True)
    selected = pd.concat(selected_frames, ignore_index=True)
    rows.to_csv(args.output / "row_graph_summary.csv", index=False)
    status.to_csv(args.output / "state_graph_summary.csv", index=False)
    selected.to_csv(args.output / "selected_specification_graph.csv", index=False)
    ordering = {value: index for index, value in enumerate(ORDER)}
    rows["order"] = rows.time_bin.map(ordering)
    status["order"] = status.time_bin.map(ordering)
    lines = [
        "# Cross-dataset state-conditioned graph discovery", "",
        "## Row-specific edge instantiation", "",
        "| Dataset | Time bin | Active P/R | Violation P/R | Topology flip |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows.sort_values(["dataset", "order"]).itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.time_bin} | {row.active_precision:.3f}/{row.active_recall:.3f} | "
            f"{row.violation_precision:.3f}/{row.violation_recall:.3f} | {row.topology_flip:.3f} |"
        )
    lines += ["", "## State-discovered batch graph", "",
              "| Dataset | Profile | Gate | Time bin | Precision | Recall |",
              "| --- | --- | --- | --- | ---: | ---: |"]
    view = status[(status.profile == "dependency") | ((status.profile == "near_hard") & (status.gate == "persistent_enabled"))]
    for row in view.sort_values(["dataset", "profile", "gate", "order"]).itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.profile} | {row.gate} | {row.time_bin} | {row.precision:.3f} | {row.recall:.3f} |"
        )
    lines += ["", "## Scope", "",
              "The experiment spans hard, near-hard, probabilistic and low-support implication candidates. It tests graph topology across states, not an exception-specific policy bank."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
