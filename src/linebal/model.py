"""Core data structures: line instances, balances and problem specs.

Operations (tasks) are numbered 1..n. Stations are numbered 1..m.
A balance maps every operation to a station.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ProblemType = Literal["SALBP-1", "SALBP-2"]


@dataclass
class Instance:
    """A single-model assembly line: operation times, precedence and zoning constraints."""

    name: str
    times: dict[int, int]
    precedences: list[tuple[int, int]] = field(default_factory=list)
    linked: list[tuple[int, int]] = field(default_factory=list)
    incompatible: list[tuple[int, int]] = field(default_factory=list)
    excluded: dict[int, list[int]] = field(default_factory=dict)  # op -> stations it may not use
    cycle_time: int | None = None
    n_stations: int | None = None

    def __post_init__(self) -> None:
        self.times = {int(k): int(v) for k, v in self.times.items()}
        self.precedences = [(int(a), int(b)) for a, b in self.precedences]
        self.linked = [(int(a), int(b)) for a, b in self.linked]
        self.incompatible = [(int(a), int(b)) for a, b in self.incompatible]
        self.excluded = {int(k): sorted(int(s) for s in v) for k, v in self.excluded.items() if v}

    @property
    def n(self) -> int:
        return len(self.times)

    @property
    def ops(self) -> list[int]:
        return sorted(self.times)

    @property
    def total_time(self) -> int:
        return sum(self.times.values())

    @property
    def max_time(self) -> int:
        return max(self.times.values())

    def predecessors(self) -> dict[int, list[int]]:
        pred: dict[int, list[int]] = {i: [] for i in self.ops}
        for a, b in self.precedences:
            pred[b].append(a)
        return {i: sorted(v) for i, v in pred.items()}

    def successors(self) -> dict[int, list[int]]:
        succ: dict[int, list[int]] = {i: [] for i in self.ops}
        for a, b in self.precedences:
            succ[a].append(b)
        return {i: sorted(v) for i, v in succ.items()}

    def topological_order(self) -> list[int]:
        """Kahn's algorithm; raises ValueError on a cycle."""
        pred_count = {i: 0 for i in self.ops}
        for _, b in self.precedences:
            pred_count[b] += 1
        succ = self.successors()
        ready = sorted(i for i, c in pred_count.items() if c == 0)
        order: list[int] = []
        while ready:
            i = ready.pop(0)
            order.append(i)
            for j in succ[i]:
                pred_count[j] -= 1
                if pred_count[j] == 0:
                    ready.append(j)
            ready.sort()
        if len(order) != self.n:
            raise ValueError(f"precedence graph of {self.name} has a cycle")
        return order

    def all_predecessors(self) -> dict[int, set[int]]:
        """Transitive closure of predecessors."""
        pred = self.predecessors()
        closure: dict[int, set[int]] = {}
        for i in self.topological_order():
            s: set[int] = set()
            for p in pred[i]:
                s.add(p)
                s |= closure[p]
            closure[i] = s
        return closure

    def all_successors(self) -> dict[int, set[int]]:
        succ = self.successors()
        closure: dict[int, set[int]] = {}
        for i in reversed(self.topological_order()):
            s: set[int] = set()
            for j in succ[i]:
                s.add(j)
                s |= closure[j]
            closure[i] = s
        return closure

    def order_strength(self) -> float:
        """Share of ordered pairs in the transitive closure (Scholl's OS, in [0, 1])."""
        n = self.n
        if n < 2:
            return 0.0
        pairs = sum(len(v) for v in self.all_successors().values())
        return pairs / (n * (n - 1) / 2)

    def has_constraints(self) -> bool:
        return bool(self.linked or self.incompatible or self.excluded)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "times": {str(k): v for k, v in sorted(self.times.items())},
            "precedences": [list(p) for p in self.precedences],
            "linked": [list(p) for p in self.linked],
            "incompatible": [list(p) for p in self.incompatible],
            "excluded": {str(k): v for k, v in sorted(self.excluded.items())},
            "cycle_time": self.cycle_time,
            "n_stations": self.n_stations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Instance:
        return cls(
            name=d["name"],
            times={int(k): v for k, v in d["times"].items()},
            precedences=[tuple(p) for p in d.get("precedences", [])],
            linked=[tuple(p) for p in d.get("linked", [])],
            incompatible=[tuple(p) for p in d.get("incompatible", [])],
            excluded={int(k): v for k, v in d.get("excluded", {}).items()},
            cycle_time=d.get("cycle_time"),
            n_stations=d.get("n_stations"),
        )

    def relabel(self, perm: dict[int, int], name: str | None = None) -> Instance:
        """Return a copy with operation ids renamed by `perm` (old -> new)."""
        return Instance(
            name=name or self.name,
            times={perm[i]: t for i, t in self.times.items()},
            precedences=[(perm[a], perm[b]) for a, b in self.precedences],
            linked=[(perm[a], perm[b]) for a, b in self.linked],
            incompatible=[(perm[a], perm[b]) for a, b in self.incompatible],
            excluded={perm[i]: s for i, s in self.excluded.items()},
            cycle_time=self.cycle_time,
            n_stations=self.n_stations,
        )


@dataclass
class Balance:
    """An assignment of operations to stations. Unassigned operations are simply absent."""

    n_stations: int
    assignment: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.assignment = {int(k): int(v) for k, v in self.assignment.items()}

    def station_ops(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {k: [] for k in range(1, self.n_stations + 1)}
        for op, st in sorted(self.assignment.items()):
            out.setdefault(st, []).append(op)
        return out

    def loads(self, inst: Instance) -> dict[int, int]:
        out = {k: 0 for k in range(1, self.n_stations + 1)}
        for op, st in self.assignment.items():
            if op in inst.times:
                out[st] = out.get(st, 0) + inst.times[op]
        return out

    def to_dict(self) -> dict:
        return {
            "n_stations": self.n_stations,
            "assignment": {str(k): v for k, v in sorted(self.assignment.items())},
        }

    @classmethod
    def from_dict(cls, d: dict) -> Balance:
        return cls(n_stations=int(d["n_stations"]), assignment={int(k): int(v) for k, v in d["assignment"].items()})


@dataclass
class ProblemSpec:
    """What the engineer actually asked for: the objective and the hard limits."""

    problem: ProblemType
    cycle_time: int | None = None  # hard limit on every station load (SALBP-1, or extra limit)
    max_stations: int | None = None  # hard limit on stations used (SALBP-2, or extra limit)

    def to_dict(self) -> dict:
        return {"problem": self.problem, "cycle_time": self.cycle_time, "max_stations": self.max_stations}

    @classmethod
    def from_dict(cls, d: dict) -> ProblemSpec:
        return cls(problem=d["problem"], cycle_time=d.get("cycle_time"), max_stations=d.get("max_stations"))
