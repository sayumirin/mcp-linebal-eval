"""Download the Scholl (1993) SALBP benchmark archive and extract the published optima.

Source: https://assembly-line-balancing.de/salbp/benchmark-data-sets-1993/
Scholl, A. (1993). Data of assembly line balancing problems. Schriften zur Quantitativen
Betriebswirtschaftslehre 16/93, TH Darmstadt. The archive also contains the Talbot et al. (1986)
and Hoffmann (1990) test sets. The site states no licence, so the files are not committed.

Outputs (all under data/, git-ignored):
  raw/SALBP-data-sets.zip             the archive (SHA-256 checked)
  raw/salbp/precedence graphs/*.IN2   precedence graphs and task times
  benchmarks/graphs.csv               graph name -> file, n, task-time sum
  benchmarks/known_optima.csv         published SALBP-1 m* and SALBP-2 c* (or [LB, UB])
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linebal.data import DATA_DIR, RAW_DIR  # noqa: E402

URL = "https://assembly-line-balancing.de/wp-content/uploads/2017/01/SALBP-data-sets.zip"
SHA256 = "cf7a5f89e034dee8fe67b305c37399920926dd174f409b96c189653ed35bb78c"


def download(dest: Path) -> None:
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == SHA256:
        print(f"{dest} already present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "mcp-linebal-eval data script"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"checksum mismatch: got {digest}")
    dest.write_bytes(data)
    print(f"downloaded {len(data)} bytes to {dest}")


def _range(cell) -> tuple[int, int] | None:
    if cell is None:
        return None
    if isinstance(cell, (int, float)):
        return int(cell), int(cell)
    m = re.match(r"\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]", str(cell))
    if m:
        return int(m.group(1)), int(m.group(2))
    if str(cell).strip().isdigit():
        v = int(str(cell).strip())
        return v, v
    return None


def extract_tables(xlsx: Path, out_dir: Path) -> None:
    import openpyxl  # dev dependency; only needed here

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs = []
    for row in wb["Precedence graphs"].iter_rows(values_only=True):
        if row and isinstance(row[0], str) and row[0].upper().endswith(".IN2"):
            graphs.append({"file": row[0].strip(), "name": str(row[1]).strip(), "n": int(row[2]), "tsum": int(row[5])})
    with (out_dir / "graphs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "file", "n", "tsum"])
        w.writeheader()
        w.writerows(graphs)

    rows = []
    for row in wb["SALBP-1 data set"].iter_rows(values_only=True):
        if not row or not isinstance(row[0], str) or not isinstance(row[1], (int, float)):
            continue
        r = _range(row[2])
        if r:
            rows.append({"graph": row[0].strip(), "problem": "SALBP-1", "given": int(row[1]), "lb": r[0], "ub": r[1]})
    for row in wb["SALBP-2 data sets"].iter_rows(values_only=True):
        if not row or not isinstance(row[0], str) or not isinstance(row[1], (int, float)):
            continue
        r = _range(row[2])
        if r:
            rows.append({"graph": row[0].strip(), "problem": "SALBP-2", "given": int(row[1]), "lb": r[0], "ub": r[1]})
    with (out_dir / "known_optima.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["graph", "problem", "given", "lb", "ub"])
        w.writeheader()
        w.writerows(rows)
    n1 = sum(r["problem"] == "SALBP-1" for r in rows)
    print(f"{len(graphs)} graphs, {n1} SALBP-1 and {len(rows) - n1} SALBP-2 published results")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    archive = DATA_DIR / "raw" / "SALBP-data-sets.zip"
    download(archive)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(RAW_DIR)
    extract_tables(RAW_DIR / "SALBP data sets.xlsx", DATA_DIR / "benchmarks")


if __name__ == "__main__":
    main()
