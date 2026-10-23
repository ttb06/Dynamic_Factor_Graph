#!/usr/bin/env python3
"""Inventory dataset assets and locally evaluated baselines on a Slurm node."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def dataset_from_path(path: Path, datasets: list[str]) -> str | None:
    text = str(path).lower().replace("-", "_")
    matches = [name for name in datasets if name.lower().replace("-", "_") in text]
    return max(matches, key=len) if matches else None


def bounded_walk(root: Path, max_depth: int = 4):
    root_depth = len(root.parts)
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.parts) - root_depth
        if depth >= max_depth:
            dirs[:] = []
        yield current_path, files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    infos: dict[str, dict] = {}
    for path in sorted((args.repo / "data").glob("*/info.json")):
        infos[path.parent.name] = json.loads(path.read_text())
    datasets = sorted(infos)
    assets: dict[str, dict[str, list[str]]] = {
        name: defaultdict(list) for name in datasets
    }

    outputs = args.analysis / "outputs"
    for directory, files in bounded_walk(outputs, max_depth=4):
        dataset = dataset_from_path(directory, datasets)
        if dataset is None:
            continue
        for filename in files:
            path = directory / filename
            lower = filename.lower()
            if lower == "config.pkl":
                assets[dataset]["configs"].append(str(path))
            elif lower.endswith(".pt") and "ema_model" in lower:
                assets[dataset]["checkpoints"].append(str(path))
            elif lower.endswith(".json") and "graph" in lower:
                assets[dataset]["graphs"].append(str(path))
            elif lower == "baseline_metrics.csv":
                assets[dataset]["metric_files"].append(str(path))
                try:
                    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                        for row in csv.DictReader(handle):
                            method = row.get("method")
                            if method:
                                assets[dataset]["methods"].append(method)
                except Exception as error:
                    assets[dataset]["metric_errors"].append(f"{path}: {error}")

    rows = []
    for name in datasets:
        info = infos[name]
        real = args.repo / "synthetic" / name / "real.csv"
        test = args.repo / "synthetic" / name / "test.csv"
        entry = assets[name]
        methods = sorted(set(entry.get("methods", [])))
        row = {
            "dataset": name,
            "task_type": info.get("task_type"),
            "target": [info.get("column_names", [])[i] for i in info.get("target_col_idx", [])],
            "train_rows": csv_rows(real) if real.exists() else None,
            "test_rows": csv_rows(test) if test.exists() else None,
            "local_baseline_methods": methods,
            "local_baseline_method_count": len(methods),
            "metric_file_count": len(entry.get("metric_files", [])),
            "config_count": len(entry.get("configs", [])),
            "checkpoint_count": len(entry.get("checkpoints", [])),
            "graph_count": len(entry.get("graphs", [])),
            "configs": sorted(set(entry.get("configs", [])))[-8:],
            "checkpoints": sorted(set(entry.get("checkpoints", [])))[-8:],
            "graphs": sorted(set(entry.get("graphs", [])))[-8:],
            "metric_files": sorted(set(entry.get("metric_files", [])))[-8:],
            "metric_errors": entry.get("metric_errors", []),
        }
        rows.append(row)

    rows.sort(key=lambda row: (
        -row["local_baseline_method_count"],
        -(row["checkpoint_count"] > 0),
        -(row["graph_count"] > 0),
        row["dataset"],
    ))
    payload = {"repo": str(args.repo), "analysis": str(args.analysis), "datasets": rows}
    (args.output / "inventory.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Multi-dataset asset inventory",
        "",
        "| dataset | task | train/test | local methods | metric files | configs | checkpoints | graphs |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['task_type']} | "
            f"{row['train_rows']}/{row['test_rows']} | "
            f"{row['local_baseline_method_count']} | {row['metric_file_count']} | "
            f"{row['config_count']} | {row['checkpoint_count']} | {row['graph_count']} |"
        )
    lines.extend(["", "## Baseline methods", ""])
    for row in rows:
        if row["local_baseline_methods"]:
            lines.append(f"- {row['dataset']}: {', '.join(row['local_baseline_methods'])}")
    (args.output / "INVENTORY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
