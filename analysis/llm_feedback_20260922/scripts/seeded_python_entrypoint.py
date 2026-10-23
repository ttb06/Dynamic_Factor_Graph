#!/usr/bin/env python3
"""Execute an upstream Python entry point after fixing all common RNGs."""
from __future__ import annotations

import argparse
import inspect
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument(
        "--extra-site", action="append", default=[],
        help="Append a compatibility site-packages directory after the active environment.")
    known, remaining = parser.parse_known_args()
    os.environ["PYTHONHASHSEED"] = str(known.seed)
    random.seed(known.seed)
    np.random.seed(known.seed)
    torch.manual_seed(known.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(known.seed)
    # Append rather than prepend: packages available in the active environment
    # retain priority, while small pure-Python compatibility dependencies can
    # be borrowed from an older official environment.
    for directory in known.extra_site:
        sys.path.append(str(Path(directory).resolve()))
    # TabSyn's 2024 code passes ``verbose`` to ReduceLROnPlateau.  Newer
    # PyTorch releases removed that no-op logging argument.  Drop it only when
    # unsupported so the optimizer and scheduling behavior remain unchanged.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    if "verbose" not in inspect.signature(scheduler.__init__).parameters:
        original_init = scheduler.__init__

        def compatible_init(self, *args, **kwargs):
            kwargs.pop("verbose", None)
            return original_init(self, *args, **kwargs)

        scheduler.__init__ = compatible_init
    sys.path.insert(0, str(Path(known.script).resolve().parent))
    sys.argv = [known.script, *remaining]
    runpy.run_path(known.script, run_name="__main__")


if __name__ == "__main__":
    main()
