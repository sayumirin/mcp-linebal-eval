import itertools
import random

import pytest

from conftest import needs_data
from linebal.cases import assemplanner_valve
from linebal.checker import check, kpis
from linebal.model import Balance, Instance, ProblemSpec
from linebal.solver import heuristic, rpw, solve
from linebal.tasks import generate_instance


def test_assemplanner_valve_optimum_is_240():
    """Acceptance criterion: the published 247 s is not optimal; 5 stations reach 240 s."""
    inst = assemplanner_valve()
    res = solve(inst, ProblemSpec("SALBP-2", max_stations=5), time_limit=20)
    assert res.status == "optimal"
    assert res.objective == 240
    assert sorted(kpis(inst, res.balance).station_loads.values()) == [205, 226, 227, 237, 240]


def test_valve_salbp1_and_infeasible_variants():
    inst = assemplanner_valve()
    assert solve(inst, ProblemSpec("SALBP-1", cycle_time=249)).objective == 5
    assert solve(inst, ProblemSpec("SALBP-1", cycle_time=239)).objective == 6
    assert solve(inst, ProblemSpec("SALBP-1", cycle_time=239, max_stations=5)).status == "infeasible"
    assert solve(inst, ProblemSpec("SALBP-1", cycle_time=146)).status == "infeasible"


def brute_force(inst: Instance, spec: ProblemSpec) -> int | None:
    """Enumerate every assignment (tiny instances only)."""
    best = None
    k_max = spec.max_stations or inst.n
    for assign in itertools.product(range(1, k_max + 1), repeat=inst.n):
        bal = Balance(k_max, dict(zip(inst.ops, assign)))
        if check(inst, bal, spec).feasible:
            k = kpis(inst, bal)
            v = k.n_stations if spec.problem == "SALBP-1" else k.cycle_time
            best = v if best is None else min(best, v)
    return best


@pytest.mark.parametrize("seed", range(8))
def test_solver_matches_brute_force_on_tiny_constrained_instances(seed):
    rng = random.Random(seed)
    inst = generate_instance(seed, 6, density=0.3, tmax=12)
    a, b, c, d = rng.sample(inst.ops, 4)
    inst.linked = [(a, b)] if seed % 2 else []
    inst.incompatible = [(c, d)]
    inst.excluded = {rng.choice(inst.ops): [1]} if seed % 3 == 0 else {}
    for spec in (ProblemSpec("SALBP-1", cycle_time=max(inst.max_time, inst.total_time // 3), max_stations=6), ProblemSpec("SALBP-2", max_stations=3)):
        expect = brute_force(inst, spec)
        res = solve(inst, spec, time_limit=10)
        if expect is None:
            assert res.status == "infeasible"
        else:
            assert res.status == "optimal" and res.objective == expect, (seed, spec, res.to_dict(), expect)


def test_rpw_respects_constraints():
    inst = generate_instance(11, 25)
    inst.linked = [(3, 4)]
    inst.incompatible = [(10, 11)]
    inst.excluded = {12: [1, 2]}
    c = max(inst.max_time, inst.total_time // 5)
    bal = rpw(inst, c)
    assert bal is not None
    assert check(inst, bal, ProblemSpec("SALBP-1", cycle_time=c)).feasible


def test_heuristic_is_an_upper_bound():
    inst = generate_instance(12, 30)
    for spec in (ProblemSpec("SALBP-1", cycle_time=inst.total_time // 6), ProblemSpec("SALBP-2", max_stations=6)):
        if spec.cycle_time is not None and spec.cycle_time < inst.max_time:
            continue
        h, e = heuristic(inst, spec), solve(inst, spec, time_limit=20)
        assert e.objective <= h.objective


@needs_data
def test_matches_published_optima_on_small_scholl_graphs():
    from linebal.data import known_optima, load_graph

    rows = [r for r in known_optima() if r["graph"] in ("Jackson", "Mitchell", "Buxey") and r["proven"]]
    assert rows
    for r in rows:
        inst = load_graph(r["graph"])
        spec = ProblemSpec("SALBP-1", cycle_time=r["given"]) if r["problem"] == "SALBP-1" else ProblemSpec("SALBP-2", max_stations=r["given"])
        res = solve(inst, spec, time_limit=20)
        assert res.status == "optimal" and res.objective == r["lb"], r
