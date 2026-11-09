"""Locate the downloaded SALBP benchmark files and the published optima.

The benchmark data is not redistributed in this repository (the source site states no licence);
run `uv run python scripts/download_data.py` first. Files then live in `data/raw/salbp/`.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

from .alb import read_instance
from .model import Instance

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("LINEBAL_DATA", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw" / "salbp"
GRAPH_DIR = RAW_DIR / "precedence graphs"
OPTIMA_CSV = DATA_DIR / "benchmarks" / "known_optima.csv"


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def graph_files() -> dict[str, Path]:
    """Map normalised graph name (e.g. 'jackson', 'weemag') to its .IN2 file."""
    if not GRAPH_DIR.exists():
        raise FileNotFoundError(f"{GRAPH_DIR} not found; run scripts/download_data.py first")
    names = {}
    names_csv = DATA_DIR / "benchmarks" / "graphs.csv"
    if names_csv.exists():
        with names_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                names[norm(row["name"])] = GRAPH_DIR / row["file"]
    for p in GRAPH_DIR.glob("*.IN2"):
        names.setdefault(norm(p.stem), p)
    return names


def load_graph(name: str) -> Instance:
    files = graph_files()
    key = norm(name)
    if key not in files:
        raise KeyError(f"unknown graph {name!r}; known: {sorted(files)}")
    inst = read_instance(files[key], name=name)
    return inst


def known_optima() -> list[dict]:
    """Rows: graph, problem (SALBP-1/SALBP-2), given, best, lb, ub, proven."""
    if not OPTIMA_CSV.exists():
        raise FileNotFoundError(f"{OPTIMA_CSV} not found; run scripts/download_data.py first")
    with OPTIMA_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("given", "lb", "ub"):
            r[k] = int(r[k])
        r["proven"] = r["lb"] == r["ub"]
    return rows
