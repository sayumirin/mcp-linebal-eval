from linebal.cases import assemplanner_valve
from linebal.checker import check, kpis
from linebal.model import Balance, Instance, ProblemSpec

OPT = {8: 1, 9: 1, 1: 2, 10: 2, 11: 3, 12: 3, 3: 4, 4: 4, 5: 4, 2: 5, 6: 5, 7: 5}


def test_optimal_valve_balance_is_feasible_with_expected_kpis():
    inst = assemplanner_valve()
    bal = Balance(5, OPT)
    res = check(inst, bal, ProblemSpec("SALBP-2", max_stations=5))
    assert res.feasible, res.violations
    k = kpis(inst, bal)
    assert k.cycle_time == 240
    assert k.station_loads == {1: 240, 2: 226, 3: 205, 4: 237, 5: 227}
    assert abs(k.line_efficiency - 1135 / 1200) < 1e-9
    assert k.idle_time == 1200 - 1135


def test_each_violation_type_is_reported():
    inst = Instance(
        "toy",
        times={1: 5, 2: 5, 3: 5, 4: 5},
        precedences=[(1, 2)],
        linked=[(3, 4)],
        incompatible=[(1, 3)],
        excluded={4: [1]},
    )
    bal = Balance(3, {1: 2, 2: 1, 3: 2, 4: 1})
    res = check(inst, bal, ProblemSpec("SALBP-1", cycle_time=8, max_stations=1))
    types = {v.type for v in res.violations}
    assert types == {"precedence", "cycle_time", "station_limit", "linked", "incompatible", "excluded"}
    over = [v for v in res.violations if v.type == "cycle_time"]
    assert all(v.amount == 2 for v in over)


def test_unassigned_and_invalid_station():
    inst = Instance("toy", times={1: 1, 2: 1})
    res = check(inst, Balance(1, {1: 3}), ProblemSpec("SALBP-1", cycle_time=5))
    assert {v.type for v in res.violations} == {"unassigned", "invalid_station"}


def test_empty_stations_are_a_warning_not_a_violation():
    inst = Instance("toy", times={1: 1, 2: 1})
    res = check(inst, Balance(3, {1: 1, 2: 3}), ProblemSpec("SALBP-2", max_stations=3))
    assert res.feasible
    assert res.warnings
    assert kpis(inst, Balance(3, {1: 1, 2: 3})).n_stations == 2
