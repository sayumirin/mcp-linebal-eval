"""Minimal .env loader (KEY=VALUE lines), so API keys can live in a git-ignored file."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path) -> list[str]:
    """Set variables from `path` that are not already in the environment. Returns the names set."""
    path = Path(path)
    if not path.exists():
        return []
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            names.append(key)
    return names
