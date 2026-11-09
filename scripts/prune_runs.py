"""Move runs that must be redone out of a log, so `run_experiment.py` reruns them on resume.

  uv run python scripts/prune_runs.py results/e1/runs.jsonl --truncated-below 65536

Removed: crashed episodes, and (with --truncated-below N) every run in which a model reply was cut
at an output limit smaller than N. Such a run was shaped by the old limit; a run whose replies never
reached the limit would have behaved the same under a higher one, so it is kept.

Removed records are appended to <log>.dropped.jsonl, never deleted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--truncated-below", type=int, default=None, help="drop runs truncated at a max_tokens below this")
    a = ap.parse_args()
    for log in a.logs:
        rows = [json.loads(line) for line in log.open(encoding="utf-8") if line.strip()]
        keep, drop = [], []
        for r in rows:
            cut = any(c.get("finish_reason") == "length" for c in r["agent"].get("llm_calls", []))
            old_cap = r["config"].get("max_tokens", 0) < (a.truncated_below or 0)
            if r.get("error") or (a.truncated_below and cut and old_cap):
                drop.append(r)
            else:
                keep.append(r)
        if drop:
            with log.with_suffix(".dropped.jsonl").open("a", encoding="utf-8", newline="\n") as f:
                for r in drop:
                    f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
            with log.open("w", encoding="utf-8", newline="\n") as f:
                for r in keep:
                    f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        n_err = sum(1 for r in drop if r.get("error"))
        print(f"{log}: kept {len(keep)}, moved {len(drop)} ({n_err} crashed, {len(drop) - n_err} truncated) to {log.with_suffix('.dropped.jsonl').name}")


if __name__ == "__main__":
    main()
