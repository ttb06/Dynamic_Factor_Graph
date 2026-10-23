#!/usr/bin/env python3
"""Load the UnoRouter credential without ever echoing it.

The project-level .env currently uses a single bare-key line.  This loader also
accepts ordinary KEY=value files so future jobs do not need shell parsing that
could leak a credential into logs or process arguments.
"""
from __future__ import annotations

import os
from pathlib import Path


DEFAULT_BASE_URL = "https://api.unorouter.com/v1"
KEY_NAMES = ("UNOROUTER_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")


def load_unorouter(env_path: str | Path) -> tuple[str, str]:
    path = Path(env_path)
    assignments: dict[str, str] = {}
    bare: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            name, value = line.split("=", 1)
            assignments[name.strip()] = value.strip().strip("\"'")
        else:
            bare.append(line)
    key = next((assignments[name] for name in KEY_NAMES if assignments.get(name)), "")
    if not key and len(bare) == 1:
        key = bare[0]
    if not key:
        raise RuntimeError(f"No UnoRouter credential found in {path}")
    base_url = assignments.get("UNOROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    return key, base_url


def install_environment(env_path: str | Path) -> str:
    key, base_url = load_unorouter(env_path)
    os.environ["UNOROUTER_API_KEY"] = key
    os.environ["UNOROUTER_BASE_URL"] = base_url
    return base_url
