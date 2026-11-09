"""Reference answers (E0).

  uv run python scripts/solve_reference.py --tasks tasks/mvp.jsonl
      exact optimum (or proof of infeasibility) for every task -> results/reference/mvp.json
  uv run python scripts/solve_reference.py --benchmark
      check the solver against published optima (Scholl 1993 data, graphs up to --max-n tasks) and
      the AssemPlanner valve case -> results/e0/benchmark.csv, results/e0/assemplanner.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from linebal.cases import ASSEMPLANNER_REPORTED, assemplanner_valve  # noqa: E402
from linebal.checker import kpis  # noqa: E402
from linebal.model import ProblemSpec  # noqa: E402
from linebal.solver import heuristic, solve  # noqa: E402
from linebal.tasks import build_instance, load_tasks  # noqa: E402


def solve_tasks(path: Path, time_limit: float) -> None:
    tasks = load_tasks(path)
    out = {}
    for t in tasks:
        inst = build_instance(t)
        spec = t.problem_spec
        res = solve(inst, spec, time_limit=time_limit)
        h = heuristic(inst, spec)
        out[t.task_id] = {
            "type": t.type,
            "problem": spec.problem,
            "status": res.status,
            "objective": res.objective,
            "lower_bound": res.lower_bound,
            "heuristic_objective": h.objective,
            "wall_time": round(res.wall_time, 3),
            "balance": res.balance.to_dict() if res.balance else None,
        }
        print(f"{t.task_id:8s} {spec.problem} {res.status:10s} obj={res.objective} rpw={h.objective} {res.wall_time:.2f}s")
    dest = ROOT / "results" / "reference" / f"{path.stem}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {dest}")


def benchmark(max_n: int, time_limit: float) -> None:
    from linebal.data import known_optima, load_graph

    rows = known_optima()
    cache = {}
    out = []
    for r in rows:
        g = r["graph"]
        if g not in cache:
            cache[g] = load_graph(g)
        inst = cache[g]
        if inst.n > max_n:
            continue
        spec = ProblemSpec("SALBP-1", cycle_time=r["given"]) if r["problem"] == "SALBP-1" else ProblemSpec("SALBP-2", max_stations=r["given"])
        res = solve(inst, spec, time_limit=time_limit)
        h = heuristic(inst, spec)
        if res.status == "optimal":
            verdict = "match" if r["proven"] and res.objective == r["lb"] else ("consistent" if r["lb"] <= res.objective <= r["ub"] else "conflict")
        elif res.objective is not None:
            verdict = "not proven" if res.objective >= r["lb"] else "conflict"
        else:
            verdict = "no solution"
        out.append(
            {
                "graph": g,
                "n": inst.n,
                "problem": r["problem"],
                "given": r["given"],
                "published_lb": r["lb"],
                "published_ub": r["ub"],
                "status": res.status,
                "objective": res.objective,
                "bound": res.lower_bound,
                "rpw": h.objective,
                "seconds": round(res.wall_time, 2),
                "verdict": verdict,
            }
        )
        print(f"{g:10s} {r['problem']} {r['given']:>5} published={r['lb']}..{r['ub']} ours={res.objective} ({res.status}) rpw={h.objective} {verdict}")
    dest = ROOT / "results" / "e0"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    # AssemPlanner valve case: 5 stations, minimise the cycle time
    inst = assemplanner_valve()
    spec = ProblemSpec("SALBP-2", max_stations=5)
    t0 = time.perf_counter()
    res = solve(inst, spec, time_limit=time_limit)
    k = kpis(inst, res.balance)
    h = heuristic(inst, spec)
    case = {
        "instance": "AssemPlanner pressure-reducing valve (Zhang et al. 2026, Table 2)",
        "stations": 5,
        "optimal_cycle_time": res.objective,
        "status": res.status,
        "station_loads": k.station_loads,
        "station_operations": res.balance.station_ops(),
        "line_balancing_rate": round(k.line_efficiency, 4),
        "rpw_cycle_time": h.objective,
        "reported": {m: {"stations": s, "cycle_time": c, "lbr": round(inst.total_time / (s * c), 4)} for m, (s, c) in ASSEMPLANNER_REPORTED.items()},
        "seconds": round(time.perf_counter() - t0, 3),
        "note": "The paper also applies a tool-change rule without giving its data; it is not modelled.",
    }
    (dest / "assemplanner.json").write_text(json.dumps(case, indent=1) + "\n", encoding="utf-8", newline="\n")
    n_match = sum(o["verdict"] == "match" for o in out)
    n_conflict = sum(o["verdict"] == "conflict" for o in out)
    print(f"{len(out)} published instances: {n_match} match, {n_conflict} conflict; valve case optimum {res.objective} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--max-n", type=int, default=60)
    ap.add_argument("--time-limit", type=float, default=60.0)
    a = ap.parse_args()
    if a.tasks:
        solve_tasks(a.tasks, a.time_limit)
    if a.benchmark:
        benchmark(a.max_n, a.time_limit)
    if not a.tasks and not a.benchmark:
        ap.error("give --tasks and/or --benchmark")


if __name__ == "__main__":
    main()
