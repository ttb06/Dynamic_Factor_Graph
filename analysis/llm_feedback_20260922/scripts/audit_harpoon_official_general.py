#!/usr/bin/env python3
"""Extract successful rows from the unmodified official Harpoon sweep."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


METHOD_MAP = {"Harpoon": "harpoon_ohe", "RePaint": "DiffPuter_Remastered"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    text = a.log.read_text(errors="replace") if a.log.exists() else ""
    completed = [(m, c) for m, c in re.findall(r"END method=(Harpoon|RePaint) constraint=(range|category|both|or) exit=0", text)]
    csv_path = a.run_root / "experiments" / "general_constraints_updated_utility.csv"
    data = pd.read_csv(csv_path)
    data = data.drop(columns=[c for c in data.columns if c.startswith("Unnamed:")], errors="ignore")
    rows, missing = [], []
    for method_label, constraint in completed:
        method = METHOD_MAP[method_label]
        match = data[(data["Dataset"].astype(str).str.lower() == "adult") & (data["Method"] == method) & (data["Constraint"] == constraint)]
        if len(match):
            rows.append(match.iloc[-1].to_dict())
        else:
            missing.append({"method": method_label, "constraint": constraint})
    out = pd.DataFrame(rows)
    out.to_csv(a.output / "official_successful_rows.csv", index=False)
    summary = {"successful_commands": len(completed), "extracted_rows": len(out), "missing_successful_rows": missing}
    (a.output / "run_summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Official Harpoon/RePaint general-constraint sweep", "",
        "Only commands whose unmodified author script exited with code 0 are included.", "",
        "| Dataset | Method | Constraint | Violation (%) | Alpha precision | KS | Detection | Identifiability |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('Dataset')} | {r.get('Method')} | {r.get('Constraint')} | {float(r.get('Avg ViolationAcc')):.4f} | "
            f"{float(r.get('Avg Alpha-P')):.4f} | {float(r.get('Avg ks')):.4f} | {float(r.get('Avg detect')):.4f} | "
            f"{float(r.get('Avg identifiability')):.4f} |"
        )
    if not rows:
        lines.append("| — | — | — | — | — | — | — | — |")
    if missing:
        lines.extend(["", "Successful commands without a matching appended CSV row: `" + json.dumps(missing) + "`."])
    (a.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(out.to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
