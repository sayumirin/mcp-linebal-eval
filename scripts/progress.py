"""Quick health check of running experiments: runs logged, crashes, truncated replies, tokens.

  uv run python scripts/progress.py results/e1/runs.jsonl results/e2/runs.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    for p in map(Path, sys.argv[1:]):
        if not p.exists():
            print(f"{p}: not started")
            continue
        rows = [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
        if not rows:
            print(f"{p}: 0 runs")
            continue
        crashes = [r["run_id"] for r in rows if r["error"]]
        cut = sum(1 for r in rows if "output_limit" in r["score"]["labels"])
        fails = Counter(r["score"]["primary_failure"] for r in rows)
        tin = sum(r["score"]["cost"]["input_tokens"] for r in rows)
        tout = sum(r["score"]["cost"]["output_tokens"] for r in rows)
        wall = sum(r["wall_time_s"] for r in rows) / len(rows)
        print(f"{p}: {len(rows)} runs, success {sum(r['score']['success'] for r in rows)}, crashes {len(crashes)}, output_limit {cut}, "
              f"tokens {tin / 1e6:.2f}M in / {tout / 1e6:.2f}M out, mean wall {wall:.0f}s, outcomes {dict(fails)}")
        for rid in crashes[:3]:
            print("   crash:", rid)


if __name__ == "__main__":
    main()
