#!/usr/bin/env python3
"""Run LCATS phase 6 with agent and evaluator in separate environments."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lcats-root", type=Path, required=True)
    parser.add_argument("--eval-python", type=Path, required=True)
    parser.add_argument("--dataset", default="insurance")
    args = parser.parse_args()

    sys.path.insert(0, str((args.lcats_root / "LCATS").resolve()))
    from phase_6 import LCATSPhase6

    # phase_6 uses sys.executable only for official evaluator subprocesses.
    # Its LLM client has already been loaded in the current agent environment.
    # Preserve the venv entrypoint path.  Resolving its symlink would change
    # sys.prefix and silently drop packages installed in that environment.
    sys.executable = str(args.eval_python)

    # The evaluator entrypoint is a compatibility venv whose Python binary
    # points into the GRDM Conda environment.  Its compiled Python extensions
    # require the matching Conda C++ runtime rather than the older system
    # libstdc++.  This environment is inherited by evaluator subprocesses.
    eval_lib = args.eval_python.resolve().parent.parent / "lib"
    old_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = str(eval_lib) + (
        f":{old_library_path}" if old_library_path else ""
    )
    asyncio.run(LCATSPhase6(data_name=args.dataset).run_reflexion_loop())


if __name__ == "__main__":
    main()
