"""Estimate the cost of a full experiment from pilot runs.

  uv run python scripts/estimate_cost.py results/pilot/runs.jsonl --price-in 0.28 --price-out 0.42 \
      --config configs/e1_verifier.yaml

Prices are per million tokens in --currency and must come from the provider's current price page
(DeepSeek lists CNY on https://api-docs.deepseek.com/zh-cn/quick_start/pricing). If the
log has cache-hit counts (DeepSeek reports them), --price-cached is used for those tokens.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from linebal.runner import read_runs  # noqa: E402
from linebal.tasks import load_tasks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--price-in", type=float, required=True, help="per 1M input tokens (cache miss)")
    ap.add_argument("--price-out", type=float, required=True, help="per 1M output tokens")
    ap.add_argument("--price-cached", type=float, default=None, help="per 1M cached input tokens")
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--config", type=Path, nargs="*", default=[], help="experiment configs to price")
    a = ap.parse_args()

    runs = read_runs(a.runs)
    per_arch: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        tin = tout = cached = 0
        for c in r["agent"].get("llm_calls", []):
            u = c.get("usage", {})
            tin += u.get("input_tokens", 0)
            tout += u.get("output_tokens", 0)
            cached += u.get("prompt_cache_hit_tokens", 0)
        price_cached = a.price_cached if a.price_cached is not None else a.price_in
        usd = ((tin - cached) * a.price_in + cached * price_cached + tout * a.price_out) / 1e6
        per_arch[r["arch"]].append(usd)
        r["_tokens"] = (tin, tout, cached)
    print(f"{len(runs)} pilot runs")
    for arch, xs in sorted(per_arch.items()):
        toks = [r["_tokens"] for r in runs if r["arch"] == arch]
        print(
            f"  {arch}: mean {sum(xs) / len(xs):.4f} {a.currency} per run, max {max(xs):.4f}; "
            f"mean tokens in {sum(t[0] for t in toks) / len(toks):,.0f} (cached {sum(t[2] for t in toks) / len(toks):,.0f}), "
            f"out {sum(t[1] for t in toks) / len(toks):,.0f}"
        )
    mean_all = {k: sum(v) / len(v) for k, v in per_arch.items()}
    for cfg_path in a.config:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        tasks = load_tasks(ROOT / cfg["tasks"])
        if cfg.get("task_types"):
            tasks = [t for t in tasks if t.type in cfg["task_types"]]
        n = len(tasks) * int(cfg.get("repeats", 3)) * len(cfg["models"])
        total = sum(n * mean_all.get(v["arch"], max(mean_all.values())) for v in cfg["variants"])
        print(f"{cfg_path.name}: {n * len(cfg['variants'])} runs, estimated {total:.2f} {a.currency}")


if __name__ == "__main__":
    main()
