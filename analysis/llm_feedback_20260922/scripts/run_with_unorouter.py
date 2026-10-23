#!/usr/bin/env python3
"""Execute a command with the project UnoRouter credential, without logging it."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from unorouter_env import load_unorouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a command is required after the options")

    key, base_url = load_unorouter(args.env_file)
    env = os.environ.copy()
    # LCATS retains OpenRouter-compatible variable names.  UnoRouter exposes
    # the same OpenAI-compatible interface, so map the newly supplied key only
    # inside this child process and never print it or pass it on argv.
    env["OPENROUTER_API_KEY"] = key
    env["OPENROUTER_BASE_URL"] = base_url
    env["UNOROUTER_API_KEY"] = key
    env["UNOROUTER_BASE_URL"] = base_url
    env["LCATS_LLM_MODEL"] = args.model
    os.execvpe(args.command[0], args.command, env)


if __name__ == "__main__":
    main()
