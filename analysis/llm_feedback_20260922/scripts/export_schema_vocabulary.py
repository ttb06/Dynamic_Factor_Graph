#!/usr/bin/env python3
"""Export categorical schema metadata only (never rows or counts)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabdiff_probe import preprocess_adult


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train, _ = preprocess_adult(args.repo)
    info = json.loads((args.repo / "data/adult/info.json").read_text())
    names = [info["column_names"][index] for index in info["target_col_idx"] + info["cat_col_idx"]]
    categories = train.cat_transform.named_steps["ordinal"].categories_
    vocabulary = {name: [str(value).strip() for value in values] for name, values in zip(names, categories)}
    args.output.write_text(json.dumps(vocabulary, indent=2))
    print(json.dumps({"columns": len(vocabulary), "values": sum(map(len, vocabulary.values())), "privacy": "schema metadata only"}, indent=2))


if __name__ == "__main__":
    main()
