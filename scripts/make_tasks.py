"""Generate a task set with a fixed seed and check every task with the exact solver.

  uv run python scripts/make_tasks.py --set mvp   # 60 tasks; needs scripts/download_data.py first
  uv run python scripts/make_tasks.py --set ci    # 10 tasks on generated lines only (no download)

Twelve tasks per type for `mvp`:
  T1  published SALBP-1 instances (cycle time given) and generated lines
  T2  published SALBP-2 instances (station count given), the AssemPlanner valve line, generated lines
  T3  T1/T2 plus one linked pair, one incompatible pair and one excluded station, chosen so that the
      unconstrained optimum violates them (the constraints bind)
  T4  infeasible, four kinds: cycle time below the longest operation; station limit below the
      minimum for the cycle time; linked pair longer than the cycle time; linked chain a-b, b-c
      with a and c incompatible
  T5  the cycle time (SALBP-1) or station count (SALBP-2) is left out of the work order
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linebal.model import ProblemSpec  # noqa: E402
from linebal.solver import solve  # noqa: E402
from linebal.tasks import Task, base_instance, build_instance, save_tasks, work_order  # noqa: E402

SEED = 2026
PUBLIC_GRAPHS = ["Jackson", "Mansoor", "Mitchell", "Roszieg", "Heskiaoff", "Buxey", "Sawyer", "Lutz1", "Gunther", "Kilbridge", "Hahn"]
GENERATED = [{"kind": "generated", "seed": s, "n": n} for s, n in ((101, 20), (102, 25), (103, 30), (104, 35))]
VALVE = {"kind": "case", "name": "assemplanner_valve"}


class Builder:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.tasks: list[Task] = []
        self.counter = {t: 0 for t in ("T1", "T2", "T3", "T4", "T5")}
        self.used_lines: set[str] = set()

    def line_name(self) -> str:
        while True:
            name = f"Line {self.rng.choice('ABCDEFGHKMNPRST')}{self.rng.randint(10, 99)}"
            if name not in self.used_lines:
                self.used_lines.add(name)
                return name

    def add(self, ttype: str, source: dict, spec: ProblemSpec, constraints: dict | None = None, hidden: dict | None = None, expected: str = "balance", note: str = "") -> Task:
        self.counter[ttype] += 1
        tid = f"{ttype}-{self.counter[ttype]:03d}"
        line = self.line_name()
        constraints = constraints or {}
        task = Task(
            task_id=tid,
            type=ttype,
            source=source,
            spec=spec.to_dict(),
            request=work_order(self.rng, line, spec, constraints, hidden),
            line_name=line,
            constraints=constraints,
            hidden=hidden or {},
            expected=expected,
            note=note,
        )
        inst = build_instance(task)
        res = solve(inst, spec, time_limit=60)
        want_infeasible = expected == "infeasible"
        if want_infeasible and res.status != "infeasible":
            raise RuntimeError(f"{tid} should be infeasible but solver says {res.status}")
        if not want_infeasible and res.status not in ("optimal", "feasible"):
            raise RuntimeError(f"{tid} should be feasible but solver says {res.status}")
        self.tasks.append(task)
        return task


def salbp1_cycle(inst, rng: random.Random, m_lo: int = 4, m_hi: int = 8) -> int:
    m = rng.randint(m_lo, m_hi)
    return max(inst.max_time, math.ceil(inst.total_time / m * rng.uniform(1.02, 1.10)))


def pick_published(rows: list[dict], graph: str, problem: str, rng: random.Random, exclude: set) -> dict:
    every = [r for r in rows if r["graph"].lower() == graph.lower() and r["problem"] == problem and r["proven"]]
    # each (graph, value) is used once while unused ones remain
    cands = [r for r in every if (graph, problem, r["given"]) not in exclude] or every
    # prefer mid-sized answers: 3..10 stations for SALBP-1, 3..10 given stations for SALBP-2
    mid = [r for r in cands if 3 <= (r["lb"] if problem == "SALBP-1" else r["given"]) <= 10]
    r = rng.choice(mid or cands)
    exclude.add((graph, problem, r["given"]))
    return r


def pick_spec(rows: list[dict], problem: str, src: dict, rng: random.Random, used: set) -> tuple[ProblemSpec, str]:
    """A cycle time (SALBP-1) or station count (SALBP-2) for this line, from the published tables
    when they cover it; otherwise chosen here and solved by our own exact solver."""
    inst = base_instance(src)
    graph = src.get("graph")
    has = graph and any(r["graph"].lower() == graph.lower() and r["problem"] == problem and r["proven"] for r in rows)
    if problem == "SALBP-1":
        if has:
            r = pick_published(rows, graph, problem, rng, used)
            return ProblemSpec("SALBP-1", cycle_time=r["given"]), f"published m*={r['lb']}"
        return ProblemSpec("SALBP-1", cycle_time=salbp1_cycle(inst, rng)), ""
    if has:
        r = pick_published(rows, graph, problem, rng, used)
        return ProblemSpec("SALBP-2", max_stations=r["given"]), f"published c*={r['lb']}"
    m = rng.randint(3, max(3, min(7, inst.n // 3)))
    return ProblemSpec("SALBP-2", max_stations=m), ""


def binding_constraints(inst, spec: ProblemSpec, rng: random.Random) -> dict:
    """One linked, one incompatible and one excluded constraint that the unconstrained optimum breaks."""
    # single worker + fixed seed: CP-SAT then returns the same optimum on every machine, so the task
    # file can be regenerated byte for byte
    base = solve(inst, spec, time_limit=60, workers=1, seed=0)
    assert base.balance is not None and base.status == "optimal"
    a = base.balance.assignment
    by_station: dict[int, list[int]] = {}
    for op, st in a.items():
        by_station.setdefault(st, []).append(op)
    # incompatible: two operations that share a station in the optimum
    st = rng.choice([s for s, ops in by_station.items() if len(ops) >= 2])
    i, j = sorted(rng.sample(by_station[st], 2))
    # linked: a precedence pair currently split over adjacent stations
    split = [(p, q) for p, q in inst.precedences if a[q] - a[p] == 1 and {p, q} & {i, j} == set()]
    lp, lq = rng.choice(split) if split else rng.choice([(p, q) for p, q in inst.precedences if {p, q} & {i, j} == set()])
    # excluded: an operation (not already constrained) barred from its current station
    free = [o for o in inst.ops if o not in {i, j, lp, lq}]
    e = rng.choice(free)
    return {"linked": [[lp, lq]], "incompatible": [[i, j]], "excluded": {str(e): [a[e]]}}


def build_mvp(rows: list[dict]) -> list[Task]:
    from linebal.data import load_graph

    b = Builder(SEED)
    rng = b.rng
    used: set = set()
    graphs = list(PUBLIC_GRAPHS)

    # T1: 9 published + 3 generated
    for g in rng.sample(graphs, 9):
        src = {"kind": "salbp", "graph": g}
        spec, note = pick_spec(rows, "SALBP-1", src, rng, used)
        b.add("T1", src, spec, note=note)
    for src in GENERATED[:3]:
        inst = base_instance(src)
        b.add("T1", src, ProblemSpec("SALBP-1", cycle_time=salbp1_cycle(inst, rng)))

    # T2: 8 published + valve + 3 generated
    for g in rng.sample(graphs, 8):
        src = {"kind": "salbp", "graph": g}
        spec, note = pick_spec(rows, "SALBP-2", src, rng, used)
        b.add("T2", src, spec, note=note)
    b.add("T2", VALVE, ProblemSpec("SALBP-2", max_stations=5), note="AssemPlanner Table 2; paper reports 247 s")
    for src in GENERATED[1:4]:
        b.add("T2", src, ProblemSpec("SALBP-2", max_stations=rng.randint(4, 7)))

    # T3: 6 on SALBP-1, 6 on SALBP-2 (public and generated), with binding constraints
    t3_sources = [{"kind": "salbp", "graph": g} for g in rng.sample(graphs, 9)] + [GENERATED[0], GENERATED[3], VALVE]
    rng.shuffle(t3_sources)
    for k, src in enumerate(t3_sources):
        inst = base_instance(src)
        spec, _ = pick_spec(rows, "SALBP-1" if k % 2 == 0 else "SALBP-2", src, rng, used)
        for attempt in range(20):
            cons = binding_constraints(inst, spec, rng)
            probe = inst.__class__.from_dict(inst.to_dict())
            probe.linked = [tuple(p) for p in cons["linked"]]
            probe.incompatible = [tuple(p) for p in cons["incompatible"]]
            probe.excluded = {int(k_): v for k_, v in cons["excluded"].items()}
            if solve(probe, spec, time_limit=30).status in ("optimal", "feasible"):
                break
        b.add("T3", src, spec, constraints=cons, note="constraints chosen to break the unconstrained optimum")

    # T4: infeasible, three of each kind
    t4_graphs = rng.sample(graphs, len(graphs)) + rng.sample(graphs, 12 - len(graphs))
    for k, g in enumerate(t4_graphs):
        src = {"kind": "salbp", "graph": g}
        inst = load_graph(g)
        kind = k % 4
        if kind == 0:
            c = inst.max_time - rng.randint(1, max(1, inst.max_time // 10))
            b.add("T4", src, ProblemSpec("SALBP-1", cycle_time=c), expected="infeasible", note=f"cycle time {c} below longest operation {inst.max_time}")
        elif kind == 1:
            r = pick_published(rows, g, "SALBP-1", rng, used)
            b.add("T4", src, ProblemSpec("SALBP-1", cycle_time=r["given"], max_stations=r["lb"] - 1), expected="infeasible", note=f"needs {r['lb']} stations at this cycle time, limit is {r['lb'] - 1}")
        elif kind == 2:
            r = pick_published(rows, g, "SALBP-1", rng, used)
            c = r["given"]
            pairs = [(p, q) for p, q in inst.precedences if inst.times[p] + inst.times[q] > c]
            if not pairs:  # pick the two longest operations if no adjacent pair is long enough
                big = sorted(inst.ops, key=lambda o: -inst.times[o])[:2]
                c = max(inst.max_time, inst.times[big[0]] + inst.times[big[1]] - 1)
                pairs = [tuple(sorted(big))]
            p, q = rng.choice(pairs)
            b.add("T4", src, ProblemSpec("SALBP-1", cycle_time=c), constraints={"linked": [[p, q]]}, expected="infeasible", note=f"linked {p},{q} take {inst.times[p] + inst.times[q]} s > cycle time {c}")
        else:
            spec, _ = pick_spec(rows, "SALBP-2", src, rng, used)
            x, y, z = sorted(rng.sample(inst.ops, 3))
            b.add("T4", src, spec, constraints={"linked": [[x, y], [y, z]], "incompatible": [[x, z]]}, expected="infeasible", note=f"linked {x}-{y}-{z} but {x},{z} incompatible")

    # T5: hidden parameter
    t5_graphs = rng.sample(graphs, 9) + [None, None, None]
    for k, g in enumerate(t5_graphs):
        src = {"kind": "salbp", "graph": g} if g else GENERATED[k - 9]
        inst = base_instance(src)
        if k % 2 == 0:
            spec, _ = pick_spec(rows, "SALBP-1", src, rng, used)
            b.add("T5", src, spec, hidden={"cycle_time": spec.cycle_time}, note="cycle time withheld")
        else:
            spec, _ = pick_spec(rows, "SALBP-2", src, rng, used)
            b.add("T5", src, spec, hidden={"max_stations": spec.max_stations}, note="station count withheld")
    return b.tasks


def build_ci() -> list[Task]:
    """Small set on generated lines and the valve case; runs without the benchmark download."""
    b = Builder(SEED + 1)
    rng = b.rng
    g20, g25, g30 = GENERATED[0], GENERATED[1], GENERATED[2]
    i20, i25 = base_instance(g20), base_instance(g25)
    b.add("T1", g20, ProblemSpec("SALBP-1", cycle_time=salbp1_cycle(i20, rng)))
    b.add("T1", VALVE, ProblemSpec("SALBP-1", cycle_time=250))
    b.add("T2", VALVE, ProblemSpec("SALBP-2", max_stations=5))
    b.add("T2", g25, ProblemSpec("SALBP-2", max_stations=5))
    spec = ProblemSpec("SALBP-1", cycle_time=salbp1_cycle(i25, rng))
    b.add("T3", g25, spec, constraints=binding_constraints(i25, spec, rng))
    spec = ProblemSpec("SALBP-2", max_stations=4)
    b.add("T3", g30, spec, constraints=binding_constraints(base_instance(g30), spec, rng))
    b.add("T4", VALVE, ProblemSpec("SALBP-1", cycle_time=140), expected="infeasible", note="cycle time 140 below longest operation 147")
    b.add("T4", VALVE, ProblemSpec("SALBP-1", cycle_time=239, max_stations=5), expected="infeasible", note="5 stations need at least 240 s")
    c = salbp1_cycle(i20, rng)
    b.add("T5", g20, ProblemSpec("SALBP-1", cycle_time=c), hidden={"cycle_time": c}, note="cycle time withheld")
    b.add("T5", VALVE, ProblemSpec("SALBP-2", max_stations=5), hidden={"max_stations": 5}, note="station count withheld")
    return b.tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["mvp", "ci"], default="mvp")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.set == "ci":
        tasks = build_ci()
    else:
        from linebal.data import known_optima

        tasks = build_mvp(known_optima())
    out = Path(a.out or Path(__file__).resolve().parents[1] / "tasks" / f"{a.set}.jsonl")
    save_tasks(tasks, out)
    counts = {}
    for t in tasks:
        counts[t.type] = counts.get(t.type, 0) + 1
    print(f"wrote {len(tasks)} tasks to {out}: {counts}")


if __name__ == "__main__":
    main()
