"""Small hand-entered instances.

`assemplanner_valve()` is the 12-operation pressure-reducing-valve line from
Zhang et al. (2026), "AssemPlanner", arXiv:2605.08831, Table 2 (assembly times in seconds and
direct predecessors). The paper balances it on 5 stations with a cycle-time target below 250 s and
reports a best cycle time of 247 s (Table 3). Its tool-change rule has no data in the paper and is
not modelled here.
"""

from __future__ import annotations

from .model import Instance

_VALVE_TIMES = {1: 124, 2: 82, 3: 71, 4: 94, 5: 72, 6: 67, 7: 78, 8: 102, 9: 138, 10: 102, 11: 147, 12: 58}
_VALVE_PRED = {2: [1], 3: [12], 5: [3, 4], 6: [5], 7: [5], 9: [8], 11: [9, 10], 12: [11]}


def assemplanner_valve() -> Instance:
    prec = [(p, j) for j, ps in _VALVE_PRED.items() for p in ps]
    return Instance(name="AssemPlanner-valve", times=dict(_VALVE_TIMES), precedences=prec, n_stations=5)


# Published results for the valve case (Table 3 of the paper): method -> (stations, cycle time s).
ASSEMPLANNER_REPORTED = {
    "IGA": (6, 234),
    "DQN": (5, 243),
    "MBO": (5, 247),
    "ABC": (5, 247),
    "ZOA": (5, 247),
    "LineBalanceAgent": (5, 247),
    "GPT-4 (w/o reflexion)": (5, 249),
}
