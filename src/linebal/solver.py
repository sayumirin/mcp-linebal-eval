"""Reference answers: an exact CP-SAT model and the ranked positional weight heuristic.

Both handle the engineering constraints used in the tasks (linked, incompatible and excluded
stations) on top of precedence and the cycle-time / station-count limits.

SALBP-1 objective: number of non-empty stations. SALBP-2 objective: realised cycle time.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from .checker import check, kpis
from .model import Balance, Instance, ProblemSpec


@dataclass
class SolveResult:
    status: str  # optimal | feasible | infeasible | unknown
    objective: int | None = None
    lower_bound: int | None = None
    balance: Balance | None = None
    wall_time: float = 0.0
    method: str = "cp-sat"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "objective": self.objective,
            "lower_bound": self.lower_bound,
            "balance": self.balance.to_dict() if self.balance else None,
            "wall_time": round(self.wall_time, 3),
            "method": self.method,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Ranked positional weight heuristic (Helgeson and Birnie 1961), constraint-aware
# ---------------------------------------------------------------------------


def _linked_groups(inst: Instance) -> dict[int, int]:
    """Union-find over linked pairs; returns op -> group representative (smallest op id)."""
    parent = {i: i for i in inst.ops}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in inst.linked:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in inst.ops}


def rpw(inst: Instance, cycle_time: int, max_index: int | None = None) -> Balance | None:
    """Station-oriented RPW for a given cycle time. Returns None if it gets stuck."""
    group_of = _linked_groups(inst)
    members: dict[int, list[int]] = {}
    for op, g in group_of.items():
        members.setdefault(g, []).append(op)
    gtime = {g: sum(inst.times[o] for o in ms) for g, ms in members.items()}
    if max(gtime.values()) > cycle_time:
        return None
    gpred: dict[int, set[int]] = {g: set() for g in members}
    for a, b in inst.precedences:
        ga, gb = group_of[a], group_of[b]
        if ga != gb:
            gpred[gb].add(ga)
    succ_all = inst.all_successors()
    weight = {}
    for g, ms in members.items():
        follow = set().union(*(succ_all[o] for o in ms)) - set(ms)
        weight[g] = gtime[g] + sum(inst.times[o] for o in follow)
    incompatible = {i: set() for i in inst.ops}
    for a, b in inst.incompatible:
        incompatible[a].add(b)
        incompatible[b].add(a)

    limit = max_index or (inst.n + max((max(s) for s in inst.excluded.values()), default=0) + 1)
    assignment: dict[int, int] = {}
    done: set[int] = set()
    k, load, station_ops = 1, 0, []
    while len(done) < len(members):
        if k > limit:
            return None
        cands = []
        for g in members:
            if g in done or not gpred[g] <= done:
                continue
            if load + gtime[g] > cycle_time:
                continue
            ms = members[g]
            if any(k in inst.excluded.get(o, ()) for o in ms):
                continue
            if any(incompatible[o] & set(station_ops) for o in ms):
                continue
            if any(incompatible[o] & set(ms) for o in ms):
                return None  # a linked group containing an incompatible pair can never be placed
            cands.append(g)
        if cands:
            g = max(cands, key=lambda x: (weight[x], -x))
            for o in members[g]:
                assignment[o] = k
                station_ops.append(o)
            load += gtime[g]
            done.add(g)
        else:
            k, load, station_ops = k + 1, 0, []
    m = max(assignment.values())
    return Balance(n_stations=m, assignment=assignment)


def heuristic(inst: Instance, spec: ProblemSpec) -> SolveResult:
    """RPW for SALBP-1; bisection on the cycle time with RPW for SALBP-2."""
    t0 = time.perf_counter()
    if spec.problem == "SALBP-1":
        assert spec.cycle_time is not None
        bal = rpw(inst, spec.cycle_time)
        if bal is not None and check(inst, bal, spec).feasible:
            obj = kpis(inst, bal).n_stations
            return SolveResult("feasible", obj, None, bal, time.perf_counter() - t0, "rpw")
        return SolveResult("unknown", None, None, None, time.perf_counter() - t0, "rpw")
    m = spec.max_stations
    assert m is not None
    lo = max(inst.max_time, math.ceil(inst.total_time / m))
    hi = inst.total_time if spec.cycle_time is None else spec.cycle_time
    best: Balance | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        bal = rpw(inst, mid, max_index=m)
        if bal is not None and check(inst, bal, ProblemSpec("SALBP-2", spec.cycle_time, m)).feasible:
            best, hi = bal, mid - 1
        else:
            lo = mid + 1
    if best is None:
        return SolveResult("unknown", None, None, None, time.perf_counter() - t0, "rpw")
    best.n_stations = m
    return SolveResult("feasible", kpis(inst, best).cycle_time, None, best, time.perf_counter() - t0, "rpw")


# ---------------------------------------------------------------------------
# Exact CP-SAT model
# ---------------------------------------------------------------------------


def solve(
    inst: Instance,
    spec: ProblemSpec,
    time_limit: float = 60.0,
    workers: int = 8,
    seed: int = 0,
) -> SolveResult:
    t0 = time.perf_counter()
    ops = inst.ops
    t = inst.times
    if spec.cycle_time is not None and spec.cycle_time < inst.max_time:
        return SolveResult("infeasible", wall_time=time.perf_counter() - t0, extra={"reason": "cycle time below longest operation"})

    heur = heuristic(inst, spec)
    q = max((max(s) for s in inst.excluded.values()), default=0)
    pred_all = inst.all_predecessors()
    succ_all = inst.all_successors()

    m = cp_model.CpModel()
    if spec.problem == "SALBP-1":
        c = spec.cycle_time
        assert c is not None
        ub = heur.objective if heur.objective is not None else inst.n
        if spec.max_stations is not None:
            ub = min(ub, spec.max_stations) if heur.objective is not None else spec.max_stations
        K = ub + q
        cap = c
    else:
        K = spec.max_stations
        assert K is not None
        cap = heur.objective if heur.objective is not None else inst.total_time
        if spec.cycle_time is not None:
            cap = min(cap, spec.cycle_time)

    # Earliest / latest station from head and tail work content (valid for any solution with
    # every load <= cap).
    lo_st = {i: max(1, math.ceil((t[i] + sum(t[p] for p in pred_all[i])) / cap)) for i in ops}
    hi_st = {i: K + 1 - math.ceil((t[i] + sum(t[s] for s in succ_all[i])) / cap) for i in ops}

    x: dict[tuple[int, int], cp_model.IntVar] = {}
    for i in ops:
        allowed = [k for k in range(lo_st[i], hi_st[i] + 1) if k not in inst.excluded.get(i, ())]
        if not allowed:
            return SolveResult("infeasible", wall_time=time.perf_counter() - t0, extra={"reason": f"operation {i} has no admissible station"})
        for k in allowed:
            x[i, k] = m.NewBoolVar(f"x_{i}_{k}")
        m.AddExactlyOne(x[i, k] for k in allowed)

    def stations_of(i: int) -> list[int]:
        return [k for (j, k) in x if j == i]

    s = {i: m.NewIntVar(lo_st[i], hi_st[i], f"s_{i}") for i in ops}
    for i in ops:
        m.Add(s[i] == sum(k * x[i, k] for k in stations_of(i)))
    for a, b in inst.precedences:
        m.Add(s[a] <= s[b])
    for a, b in inst.linked:
        m.Add(s[a] == s[b])
    for a, b in inst.incompatible:
        for k in set(stations_of(a)) & set(stations_of(b)):
            m.AddBoolOr([x[a, k].Not(), x[b, k].Not()])

    load = {k: sum(t[i] * x[i, kk] for (i, kk) in x if kk == k) for k in range(1, K + 1)}

    if spec.problem == "SALBP-1":
        z = {k: m.NewBoolVar(f"z_{k}") for k in range(1, K + 1)}
        for (i, k), var in x.items():
            m.AddImplication(var, z[k])
        for k in range(1, K + 1):
            m.Add(load[k] <= cap * z[k])
        if q == 0:  # stations are interchangeable only without position-specific exclusions
            for k in range(1, K):
                m.Add(z[k] >= z[k + 1])
        n_used = sum(z.values())
        lb = math.ceil(inst.total_time / cap)
        m.Add(n_used >= lb)
        if spec.max_stations is not None:
            m.Add(n_used <= spec.max_stations)
        m.Minimize(n_used)
    else:
        lb = max(inst.max_time, math.ceil(inst.total_time / K))
        ct = m.NewIntVar(lb, cap, "ct")
        for k in range(1, K + 1):
            m.Add(load[k] <= ct)
        m.Minimize(ct)

    if heur.balance is not None:
        for (i, k), var in x.items():
            m.AddHint(var, 1 if heur.balance.assignment.get(i) == k else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    st = solver.Solve(m)
    wall = time.perf_counter() - t0
    if st == cp_model.INFEASIBLE:
        return SolveResult("infeasible", wall_time=wall)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return SolveResult("unknown", lower_bound=int(solver.BestObjectiveBound()) if st != cp_model.MODEL_INVALID else None, wall_time=wall)
    assignment = {i: k for (i, k), var in x.items() if solver.Value(var)}
    n_st = K if spec.problem == "SALBP-2" else max(assignment.values())
    bal = Balance(n_stations=n_st, assignment=assignment)
    res = check(inst, bal, spec)
    if not res.feasible:  # should never happen; guards against modelling bugs
        raise RuntimeError(f"CP-SAT returned an infeasible balance: {res.violations[:3]}")
    k_ = kpis(inst, bal)
    obj = k_.n_stations if spec.problem == "SALBP-1" else k_.cycle_time
    return SolveResult(
        "optimal" if st == cp_model.OPTIMAL else "feasible",
        obj,
        int(round(solver.BestObjectiveBound())),
        bal,
        wall,
        "cp-sat",
        {"heuristic_objective": heur.objective},
    )
