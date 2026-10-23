#!/usr/bin/env python3
"""Build a Git-ready research snapshot without datasets, checkpoints, or secrets.

The bundle mirrors the source repository, adds navigation documentation, and
curates experiment evidence plus logs. Large data/model/sample artifacts remain
on the research server and are indexed in an omission manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


BINARY_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".npy", ".npz",
    ".parquet", ".feather", ".h5", ".hdf5", ".zip", ".tar", ".gz",
    ".7z", ".rar", ".png", ".jpg", ".jpeg", ".pdf",
}
CODE_SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "node_modules", "data", "synthetic",
    "outputs", "logs", "wandb", "checkpoints", "package_exports",
}
SAMPLE_NAMES = {
    "selected.csv", "b_reference.csv", "g_reference.csv", "uniform_control.csv",
    "dense_factor.csv", "dense_factor_then_certifier.csv", "final_repair.csv",
    "unconditional.csv", "synthetic.csv", "real.csv",
}
CSV_EVIDENCE_MARKERS = {
    "metric", "summary", "per_seed", "winner", "audit", "record", "diagnostic",
    "manifest", "selected_indices", "candidate", "mechanism", "constraint",
}
SECRET_PATTERNS = (
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "<REDACTED_OPENAI_KEY>"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "<REDACTED_GITHUB_TOKEN>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<REDACTED_AWS_KEY>"),
    (re.compile(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]{16,}"), "Bearer <REDACTED>"),
    (re.compile(
        r"(?m)^([A-Z0-9_]*(?:API_KEY|TOKEN|PASSWORD|SECRET)[A-Z0-9_]*)=(.+)$"),
     r"\1=<REDACTED>"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redact_text(data: bytes) -> tuple[bytes, int]:
    text = data.decode("utf-8", errors="replace")
    changes = 0
    for pattern, replacement in SECRET_PATTERNS:
        text, count = pattern.subn(replacement, text)
        changes += count
    return text.encode("utf-8"), changes


def copy_text_redacted(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    data, changes = redact_text(source.read_bytes())
    target.write_bytes(data)
    shutil.copystat(source, target, follow_symlinks=True)
    return changes


def is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in BINARY_SUFFIXES:
        return False
    try:
        with path.open("rb") as handle:
            sample = handle.read(8192)
    except OSError:
        return False
    return b"\x00" not in sample


def run_capture(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command, cwd=cwd, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-evidence-mb", type=float, default=20.0)
    parser.add_argument("--max-full-log-mb", type=float, default=8.0)
    args = parser.parse_args()

    repo = args.repo.resolve()
    analysis = args.analysis.resolve()
    docs_source = args.docs.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = output_dir / args.name
    archive = output_dir / f"{args.name}.zip"
    if bundle.exists() or archive.exists():
        raise FileExistsError(f"output already exists: {bundle} or {archive}")
    bundle.mkdir()

    omitted: list[dict[str, object]] = []
    redactions: list[tuple[str, int]] = []

    def omit(path: Path, category: str, reason: str) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        omitted.append({
            "category": category,
            "original_path": str(path),
            "size_bytes": size,
            "reason": reason,
        })

    def copy_file(source: Path, target: Path, category: str) -> None:
        if source.name == ".env" or source.name.startswith(".env."):
            omit(source, category, "secret_environment_file")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if is_probably_text(source):
            changes = copy_text_redacted(source, target)
            if changes:
                redactions.append((str(target.relative_to(bundle)), changes))
        else:
            shutil.copy2(source, target)

    # Mirror repository code, excluding analysis/data/model/generated artifacts.
    for root, dirs, files in os.walk(repo):
        root_path = Path(root)
        relative_root = root_path.relative_to(repo)
        if relative_root.parts and relative_root.parts[0] == "analysis":
            dirs[:] = []
            continue
        dirs[:] = sorted(d for d in dirs if d not in CODE_SKIP_DIRS)
        for filename in sorted(files):
            source = root_path / filename
            relative = source.relative_to(repo)
            if source.suffix.lower() in BINARY_SUFFIXES:
                omit(source, "repo", "binary_or_archive_excluded")
                continue
            if source.stat().st_size > 25 * 1024 * 1024:
                omit(source, "repo", "source_file_over_25MB")
                continue
            copy_file(source, bundle / relative, "repo")

    # Preserve the upstream README while installing a research-oriented entry point.
    upstream = bundle / "README.md"
    if upstream.exists():
        upstream.replace(bundle / "README_UPSTREAM.md")

    # Copy research code and launchers in their original relative locations.
    analysis_target = bundle / "analysis" / analysis.name
    for dirname in ("scripts", "slurm"):
        source_dir = analysis / dirname
        if not source_dir.exists():
            continue
        for source in sorted(source_dir.rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            if source.suffix.lower() in BINARY_SUFFIXES:
                omit(source, "analysis_code", "binary_or_archive_excluded")
                continue
            copy_file(source, analysis_target / dirname / source.relative_to(source_dir),
                      "analysis_code")

    # Top-level ledgers/reports in the analysis directory.
    for source in sorted(analysis.iterdir()):
        if not source.is_file() or source.name == ".env":
            continue
        if source.suffix.lower() in {".md", ".txt", ".csv", ".json"}:
            copy_file(source, analysis_target / source.name, "analysis_top_level")

    # Curate output evidence while excluding generated tables and model artifacts.
    evidence_root = analysis_target / "evidence"
    outputs_root = analysis / "outputs"
    max_evidence = int(args.max_evidence_mb * 1024 * 1024)
    if outputs_root.exists():
        for source in sorted(outputs_root.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(outputs_root)
            suffix = source.suffix.lower()
            lower_name = source.name.lower()
            if "samples" in relative.parts or lower_name in SAMPLE_NAMES:
                omit(source, "output", "generated_sample_or_pool")
                continue
            if suffix in BINARY_SUFFIXES:
                omit(source, "output", "checkpoint_array_or_archive")
                continue
            include = False
            if suffix in {".md", ".txt", ".pointer"}:
                include = True
            elif suffix == ".json":
                include = True
            elif suffix == ".csv":
                include = any(marker in lower_name for marker in CSV_EVIDENCE_MARKERS)
            if not include:
                omit(source, "output", "non_evidence_table_or_unknown_format")
                continue
            if source.stat().st_size > max_evidence:
                omit(source, "output", f"evidence_file_over_{args.max_evidence_mb:g}MB")
                continue
            copy_file(source, evidence_root / "outputs" / relative, "output_evidence")

    # Preserve all small logs; truncate only unusually large individual files.
    logs_root = analysis / "logs"
    max_full_log = int(args.max_full_log_mb * 1024 * 1024)
    if logs_root.exists():
        for source in sorted(logs_root.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(logs_root)
            target = evidence_root / "logs" / relative
            size = source.stat().st_size
            if size <= max_full_log:
                copy_file(source, target, "log")
                continue
            with source.open("rb") as handle:
                head = handle.read(512 * 1024)
                handle.seek(max(0, size - 2 * 1024 * 1024))
                tail = handle.read()
            combined = (
                head + b"\n\n[... LOG TRUNCATED IN GIT SNAPSHOT; SEE MANIFEST ...]\n\n" + tail)
            target = target.with_name(target.name + ".headtail.txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            redacted, changes = redact_text(combined)
            target.write_bytes(redacted)
            if changes:
                redactions.append((str(target.relative_to(bundle)), changes))
            omit(source, "log", f"full_log_over_{args.max_full_log_mb:g}MB_headtail_included")

    # Install human/agent navigation documents.
    docs_target = bundle / "docs"
    docs_target.mkdir(exist_ok=True)
    for source in sorted(docs_source.iterdir()):
        if not source.is_file():
            continue
        if source.name == "README.md":
            copy_file(source, bundle / "README.md", "documentation")
        else:
            copy_file(source, docs_target / source.name, "documentation")

    # Git state and environment snapshot.
    manifests = bundle / "manifests"
    manifests.mkdir(exist_ok=True)
    (manifests / "GIT_COMMIT.txt").write_text(
        run_capture(["git", "log", "-1", "--format=fuller"], repo))
    (manifests / "GIT_STATUS.txt").write_text(
        run_capture(["git", "status", "--short"], repo))
    (manifests / "UNCOMMITTED_CHANGES.patch").write_text(
        run_capture(["git", "diff", "--binary"], repo))
    (manifests / "UNTRACKED_ANALYSIS_NOTE.md").write_text(
        "The source repository reported `analysis/` as untracked. This snapshot deliberately "
        "includes its scripts, Slurm launchers, evidence, and logs without altering the source checkout.\n")

    environment = bundle / "environment"
    environment.mkdir(exist_ok=True)
    (environment / "python_version.txt").write_text(sys.version + "\n")
    (environment / "pip_freeze.txt").write_text(
        run_capture([sys.executable, "-m", "pip", "freeze"], repo))

    gitignore = """# Research data and generated artifacts
.env
.env.*
data/
synthetic/
outputs/
logs/
checkpoints/
*.pt
*.pth
*.ckpt
*.pkl
*.pickle
*.npy
*.npz
*.parquet
*.zip
*.tar
*.tar.gz
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
wandb/
"""
    (bundle / ".gitignore").write_text(gitignore)

    with (manifests / "OMITTED_ARTIFACTS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "category", "original_path", "size_bytes", "reason"))
        writer.writeheader()
        writer.writerows(omitted)

    with (manifests / "REDACTIONS.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("packaged_path", "redaction_count"))
        writer.writerows(redactions)

    # Manifest every packaged file before archiving.
    packaged_rows = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "FILE_MANIFEST.csv":
            packaged_rows.append({
                "relative_path": str(path.relative_to(bundle)).replace(os.sep, "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    with (manifests / "FILE_MANIFEST.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("relative_path", "size_bytes", "sha256"))
        writer.writeheader()
        writer.writerows(packaged_rows)

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9, allowZip64=True) as handle:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                handle.write(path, Path(args.name) / path.relative_to(bundle))

    total_size = sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())
    summary = output_dir / f"{args.name}_PACKAGE_SUMMARY.md"
    summary.write_text(
        "# Package summary\n\n"
        f"- Created UTC: `{datetime.now(timezone.utc).isoformat()}`\n"
        f"- Bundle: `{bundle}`\n"
        f"- ZIP: `{archive}`\n"
        f"- ZIP SHA-256: `{sha256(archive)}`\n"
        f"- Packaged files: `{len(packaged_rows) + 1}`\n"
        f"- Uncompressed bytes: `{total_size}`\n"
        f"- ZIP bytes: `{archive.stat().st_size}`\n"
        f"- Omitted/indexed artifacts: `{len(omitted)}`\n"
        f"- Redacted secret-like values: `{sum(count for _, count in redactions)}`\n")
    print(summary)
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

