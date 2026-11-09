"""Constraint checking and KPIs. The MCP `validate_balance` tool, the A2 verifier and the scorer
all call this module, so there is exactly one definition of "feasible"."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .model import Balance, Instance, ProblemSpec

VIOLATION_TYPES = (
    "unassigned",
    "invalid_station",
    "precedence",
    "cycle_time",
    "station_limit",
    "linked",
    "incompatible",
    "excluded",
)
# Constraint families used for per-type reporting.
ENGINEERING_TYPES = ("linked", "incompatible", "excluded")


@dataclass
class Violation:
    type: str
    message: str
    operations: list[int] = field(default_factory=list)
    stations: list[int] = field(default_factory=list)
    amount: int | None = None  # e.g. seconds over the cycle time

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, [])}


@dataclass
class CheckResult:
    feasible: bool
    violations: list[Violation]
    warnings: list[str]

    def counts(self) -> dict[str, int]:
        out = {t: 0 for t in VIOLATION_TYPES}
        for v in self.violations:
            out[v.type] += 1
        return out

    def to_dict(self, limit: int | None = None) -> dict:
        vs = self.violations if limit is None else self.violations[:limit]
        d = {
            "feasible": self.feasible,
            "n_violations": len(self.violations),
            "violations": [v.to_dict() for v in vs],
        }
        if limit is not None and len(self.violations) > limit:
            d["truncated"] = f"showing {limit} of {len(self.violations)} violations"
        if self.warnings:
            d["warnings"] = self.warnings
        return d


def check(inst: Instance, bal: Balance, spec: ProblemSpec) -> CheckResult:
    a = bal.assignment
    vs: list[Violation] = []
    warnings: list[str] = []

    missing = [i for i in inst.ops if i not in a]
    if missing:
        vs.append(
            Violation(
                "unassigned",
                f"{len(missing)} operation(s) are not assigned to any station: {missing}",
                operations=missing,
            )
        )
    for op, st in sorted(a.items()):
        if op not in inst.times:
            vs.append(Violation("invalid_station", f"Operation {op} does not exist", operations=[op]))
        elif not 1 <= st <= bal.n_stations:
            vs.append(
                Violation(
                    "invalid_station",
                    f"Operation {op} is on station {st}, but the line has stations 1..{bal.n_stations}",
                    operations=[op],
                    stations=[st],
                )
            )

    for i, j in inst.precedences:
        if i in a and j in a and a[i] > a[j]:
            vs.append(
                Violation(
                    "precedence",
                    f"Operation {j} (station {a[j]}) must not be upstream of its predecessor {i} (station {a[i]})",
                    operations=[i, j],
                    stations=[a[i], a[j]],
                )
            )

    loads = bal.loads(inst)
    if spec.cycle_time is not None:
        for st, load in sorted(loads.items()):
            if load > spec.cycle_time:
                vs.append(
                    Violation(
                        "cycle_time",
                        f"Station {st} load {load} exceeds the cycle time {spec.cycle_time} by {load - spec.cycle_time}",
                        operations=sorted(op for op, s in a.items() if s == st),
                        stations=[st],
                        amount=load - spec.cycle_time,
                    )
                )

    used = sorted({st for op, st in a.items() if op in inst.times and 1 <= st <= bal.n_stations})
    if spec.max_stations is not None and len(used) > spec.max_stations:
        vs.append(
            Violation(
                "station_limit",
                f"{len(used)} stations are in use, but at most {spec.max_stations} are allowed",
                stations=used,
                amount=len(used) - spec.max_stations,
            )
        )
    empty = [k for k in range(1, bal.n_stations + 1) if k not in used]
    if empty and a:
        warnings.append(f"Station(s) {empty} have no operations")

    for i, j in inst.linked:
        if i in a and j in a and a[i] != a[j]:
            vs.append(
                Violation(
                    "linked",
                    f"Operations {i} and {j} must share a station but are on stations {a[i]} and {a[j]}",
                    operations=[i, j],
                    stations=[a[i], a[j]],
                )
            )
    for i, j in inst.incompatible:
        if i in a and j in a and a[i] == a[j]:
            vs.append(
                Violation(
                    "incompatible",
                    f"Operations {i} and {j} must not share a station but are both on station {a[i]}",
                    operations=[i, j],
                    stations=[a[i]],
                )
            )
    for i, banned in sorted(inst.excluded.items()):
        if i in a and a[i] in banned:
            vs.append(
                Violation(
                    "excluded",
                    f"Operation {i} may not be placed on station {a[i]} (excluded stations: {banned})",
                    operations=[i],
                    stations=[a[i]],
                )
            )

    return CheckResult(feasible=not vs, violations=vs, warnings=warnings)


@dataclass
class KPIs:
    n_stations: int  # stations with at least one operation
    cycle_time: int  # realised cycle time = max station load
    line_efficiency: float  # total work content / (n_stations * cycle_time)
    idle_time: int  # n_stations * cycle_time - total work content
    smoothness_index: float
    station_loads: dict[int, int]
    complete: bool  # every operation assigned

    def to_dict(self) -> dict:
        d = asdict(self)
        d["line_efficiency"] = round(self.line_efficiency, 4)
        d["smoothness_index"] = round(self.smoothness_index, 2)
        d["station_loads"] = {str(k): v for k, v in sorted(self.station_loads.items())}
        return d


def kpis(inst: Instance, bal: Balance) -> KPIs:
    loads = {k: v for k, v in bal.loads(inst).items() if 1 <= k <= bal.n_stations}
    used = {k: v for k, v in loads.items() if v > 0}
    m = len(used)
    ct = max(used.values()) if used else 0
    work = sum(used.values())
    eff = work / (m * ct) if m and ct else 0.0
    smooth = sum((ct - v) ** 2 for v in used.values()) ** 0.5 if used else 0.0
    complete = all(i in bal.assignment for i in inst.ops)
    return KPIs(
        n_stations=m,
        cycle_time=ct,
        line_efficiency=eff,
        idle_time=m * ct - work,
        smoothness_index=smooth,
        station_loads=loads,
        complete=complete,
    )


def objective_value(spec: ProblemSpec, k: KPIs) -> int:
    """SALBP-1 minimises stations, SALBP-2 minimises the cycle time."""
    return k.n_stations if spec.problem == "SALBP-1" else k.cycle_time
