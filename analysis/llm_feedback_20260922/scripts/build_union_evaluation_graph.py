#!/usr/bin/env python3
"""Build a frozen union audit graph for fair comparison of graph portfolios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from curate_expanded_constraint_catalog import exact_program_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selected, seen, duplicate_rows = [], {}, []
    for origin, path in (("old", args.old), ("new", args.new)):
        for edge in json.loads(path.read_text()):
            key = exact_program_key(edge)
            if key in seen:
                duplicate_rows.append({"origin": origin, "candidate_id": edge["candidate_id"], "duplicate_of": seen[key]})
                continue
            item = dict(edge)
            item["evaluation_origin"] = origin
            item["original_candidate_id"] = edge["candidate_id"]
            item["candidate_id"] = f"{origin}::{edge['candidate_id']}"
            selected.append(item)
            seen[key] = item["candidate_id"]
    (args.output/"evaluation_graph.json").write_text(json.dumps(selected, indent=2))
    (args.output/"duplicates.json").write_text(json.dumps(duplicate_rows, indent=2))
    summary = {"old_edges": len(json.loads(args.old.read_text())), "new_edges": len(json.loads(args.new.read_text())),
               "union_edges": len(selected), "deduplicated": len(duplicate_rows),
               "selection_uses_heldout": False}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
