#!/usr/bin/env python3
"""Download and prepare selected official TabDiff datasets."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", default=["magic", "diabetes", "beijing", "news"],
        choices=["magic", "diabetes", "beijing", "news"],
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from download_dataset import download_from_uci

    for dataset in args.datasets:
        download_from_uci(dataset)
        subprocess.run(
            [sys.executable, "process_dataset.py", "--dataname", dataset],
            cwd=args.repo,
            check=True,
        )


if __name__ == "__main__":
    main()
