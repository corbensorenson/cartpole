from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "sha256": file_sha256(path),
    }


def data_sha256(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {
            "mujoco": _package_version("mujoco"),
            "mlx": _package_version("mlx"),
            "numpy": _package_version("numpy"),
            "gymnasium": _package_version("gymnasium"),
        },
    }


def git_metadata(cwd: str | Path) -> dict[str, Any]:
    cwd = Path(cwd)

    def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    root = run_git(["rev-parse", "--show-toplevel"])
    if root.returncode != 0:
        return {"available": False}

    repo_root = Path(root.stdout.strip())

    def run_root_git(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    head = run_root_git(["rev-parse", "HEAD"])
    status = run_root_git(["status", "--short"])
    commit = head.stdout.strip() if head.returncode == 0 else None
    short_status = status.stdout.splitlines() if status.returncode == 0 else []
    return {
        "available": True,
        "root": str(repo_root),
        "commit": commit,
        "dirty": bool(short_status),
        "status_short": short_status,
    }
