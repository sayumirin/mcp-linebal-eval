"""Run an experiment from a YAML config and append run records to its JSONL log.

  uv run python scripts/run_experiment.py configs/e1_verifier.yaml
  uv run python scripts/run_experiment.py configs/e1_verifier.yaml --limit 5      # pilot on 5 tasks
  uv run python scripts/run_experiment.py configs/ci_smoke.yaml --out /tmp/x.jsonl

Runs already present in the log are skipped, so an interrupted experiment can be resumed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from linebal.agents.base import RunConfig  # noqa: E402
from linebal.agents.llm import make_llm  # noqa: E402
from linebal.faults import FaultConfig  # noqa: E402
from linebal.runner import run_experiment  # noqa: E402
from linebal.envfile import load_env_file  # noqa: E402
from linebal.tasks import load_tasks  # noqa: E402


def build_variants(cfg: dict, tasks_path: str) -> list[RunConfig]:
    defaults = cfg.get("defaults", {})
    out = []
    for v in cfg["variants"]:
        merged = {**defaults, **v}
        f = merged.pop("faults", {}) or {}
        out.append(
            RunConfig(
                arch=merged["arch"],
                escalation=merged.get("escalation", "lenient"),
                temperature=float(merged.get("temperature", 0.0)),
                max_tokens=int(merged.get("max_tokens", 4096)),
                max_steps=int(merged.get("max_steps", 50)),
                max_rounds=int(merged.get("max_rounds", 10)),
                transport=merged.get("transport", "inproc"),
                faults=FaultConfig(rate=float(f.get("rate", 0.0)), kinds=list(f.get("kinds", ["transient"]))),
                tasks_path=tasks_path,
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks (after the type filter)")
    ap.add_argument("--tasks", nargs="*", default=None, help="only these task ids")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=None, help="override the config's repeats (e.g. 1 for a pilot)")
    ap.add_argument("--dry-run", action="store_true", help="print the job count and exit")
    a = ap.parse_args()

    load_env_file(ROOT / ".env")  # API keys may live in a git-ignored .env
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    tasks_path = str(ROOT / cfg["tasks"])
    tasks = load_tasks(tasks_path)
    types = cfg.get("task_types")
    if types:
        tasks = [t for t in tasks if t.type in types]
    if a.tasks:
        tasks = [t for t in tasks if t.task_id in set(a.tasks)]
    if a.limit:
        # spread a pilot over the task types rather than taking the first N of one type
        by_type: dict[str, list] = {}
        for t in tasks:
            by_type.setdefault(t.type, []).append(t)
        picked = []
        while len(picked) < a.limit and any(by_type.values()):
            for ty in sorted(by_type):
                if by_type[ty] and len(picked) < a.limit:
                    picked.append(by_type[ty].pop(0))
        tasks = picked
    ref_path = ROOT / "results" / "reference" / f"{Path(cfg['tasks']).stem}.json"
    references = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {}
    if not references:
        print(f"warning: no reference answers at {ref_path}; run scripts/solve_reference.py first", file=sys.stderr)
    variants = build_variants(cfg, cfg["tasks"])  # logged as written in the config, relative to the repo root
    repeats = a.repeats or int(cfg.get("repeats", 3))
    n_jobs = len(cfg["models"]) * len(variants) * len(tasks) * repeats
    print(f"{cfg['experiment']}: {len(tasks)} tasks x {len(variants)} variants x {len(cfg['models'])} models x {repeats} repeats = {n_jobs} runs")
    if a.dry_run:
        return
    models = [(m.get("label", m.get("model", m.get("provider"))), make_llm(m)) for m in cfg["models"]]
    out = a.out or ROOT / cfg["out"]
    asyncio.run(
        run_experiment(
            cfg["experiment"],
            tasks,
            references,
            models,
            variants,
            repeats,
            Path(out),
            concurrency=a.concurrency or int(cfg.get("concurrency", 4)),
        )
    )


if __name__ == "__main__":
    main()
