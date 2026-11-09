"""Task definitions: a natural-language work order on top of a line instance.

Types
  T1  cycle time given, minimise stations (SALBP-1)
  T2  station count given, minimise cycle time (SALBP-2)
  T3  T1 or T2 plus engineering constraints (linked / incompatible / excluded stations)
  T4  infeasible: the requirements contradict each other; the right answer is to say so
  T5  underspecified: the key parameter is missing; the right answer is to ask the engineer first

A task stores only a reference to its line data (benchmark graph name, built-in case or generator
seed) plus the modifications, so public benchmark data never has to be committed.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .cases import assemplanner_valve
from .model import Instance, ProblemSpec

TASK_TYPES = ("T1", "T2", "T3", "T4", "T5")


@dataclass
class Task:
    task_id: str
    type: str
    source: dict  # {"kind": "salbp", "graph": ...} | {"kind": "case", "name": ...} | {"kind": "generated", ...}
    spec: dict  # full ProblemSpec, including values hidden from the agent
    request: str  # the work order text given to the agent
    line_name: str
    constraints: dict = field(default_factory=dict)  # linked / incompatible / excluded
    hidden: dict = field(default_factory=dict)  # T5: spec fields withheld until the engineer is asked
    expected: str = "balance"  # "balance" or "infeasible"
    note: str = ""  # why the task is infeasible, how it was built, ...
    relabel_seed: int | None = None

    @property
    def problem_spec(self) -> ProblemSpec:
        return ProblemSpec.from_dict(self.spec)

    def visible_spec(self, revealed: bool = False) -> ProblemSpec:
        d = dict(self.spec)
        if not revealed:
            for k in self.hidden:
                d[k] = None
        return ProblemSpec.from_dict(d)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Task:
        return cls(**d)


def generate_instance(seed: int, n: int, density: float = 0.18, tmin: int = 2, tmax: int = 40) -> Instance:
    """Random precedence graph (numbered topologically) with integer task times.

    Used for fresh instances that cannot appear in any model's training data.
    """
    rng = random.Random(seed)
    prec = set()
    for j in range(2, n + 1):
        for i in range(max(1, j - 8), j):
            if rng.random() < density:
                prec.add((i, j))
    # transitive reduction so only direct relations are stored (as the ALB format asks)
    inst = Instance(name=f"GEN-{seed}", times={i: 1 for i in range(1, n + 1)}, precedences=sorted(prec))
    succ_all = inst.all_successors()
    direct = []
    for i, j in sorted(prec):
        if not any(j in succ_all[k] for k in inst.successors()[i] if k != j):
            direct.append((i, j))
    # skewed time distribution: many short operations, a few long ones
    times = {i: max(tmin, min(tmax, int(rng.lognormvariate(2.3, 0.6)))) for i in range(1, n + 1)}
    return Instance(name=f"GEN-{seed}", times=times, precedences=direct)


def base_instance(source: dict) -> Instance:
    kind = source["kind"]
    if kind == "salbp":
        from .data import load_graph

        return load_graph(source["graph"])
    if kind == "case":
        if source["name"] == "assemplanner_valve":
            return assemplanner_valve()
        raise KeyError(source["name"])
    if kind == "generated":
        return generate_instance(source["seed"], source["n"], source.get("density", 0.18))
    raise KeyError(kind)


def relabel_permutation(n: int, seed: int) -> dict[int, int]:
    rng = random.Random(seed)
    new = list(range(1, n + 1))
    rng.shuffle(new)
    return {old: new[old - 1] for old in range(1, n + 1)}


def build_instance(task: Task) -> Instance:
    inst = base_instance(task.source)
    if task.relabel_seed is not None:
        inst = inst.relabel(relabel_permutation(inst.n, task.relabel_seed))
    c = task.constraints
    inst.linked = [tuple(p) for p in c.get("linked", [])]
    inst.incompatible = [tuple(p) for p in c.get("incompatible", [])]
    inst.excluded = {int(k): list(v) for k, v in c.get("excluded", {}).items()}
    inst.name = task.line_name
    spec = task.problem_spec
    inst.cycle_time = spec.cycle_time
    inst.n_stations = spec.max_stations
    return inst


def load_tasks(path: str | Path) -> list[Task]:
    with Path(path).open(encoding="utf-8") as f:
        return [Task.from_dict(json.loads(line)) for line in f if line.strip()]


def save_tasks(tasks: list[Task], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Work-order text
# ---------------------------------------------------------------------------


def describe_constraints(constraints: dict) -> list[str]:
    lines = []
    for a, b in constraints.get("linked", []):
        lines.append(f"operations {a} and {b} use the same fixture and must be on the same station")
    for a, b in constraints.get("incompatible", []):
        lines.append(f"operations {a} and {b} must not be on the same station")
    for op, st in sorted(constraints.get("excluded", {}).items(), key=lambda kv: int(kv[0])):
        sts = ", ".join(str(s) for s in st)
        lines.append(f"operation {op} cannot be done at station{'s' if len(st) > 1 else ''} {sts}")
    return lines


def work_order(
    rng: random.Random,
    line: str,
    spec: ProblemSpec,
    constraints: dict | None = None,
    hidden: dict | None = None,
) -> str:
    hidden = hidden or {}
    wo = f"WO-{rng.randint(1000, 9999)}"
    parts = [f"Work order {wo} for {line}."]
    if spec.problem == "SALBP-1":
        if "cycle_time" in hidden:
            parts.append(
                rng.choice(
                    [
                        "The line is being re-balanced for next quarter's volume. Please produce a new balance.",
                        "Demand is changing, so the line needs a new balance. Please set it up.",
                    ]
                )
            )
        else:
            parts.append(
                rng.choice(
                    [
                        f"The line must run at a cycle time of {spec.cycle_time} s. Assign every operation to a station so that no station exceeds the cycle time, using as few stations as possible.",
                        f"Target cycle time is {spec.cycle_time} s. Balance the line with the minimum number of stations; no station may be loaded above the cycle time.",
                    ]
                )
            )
            if spec.max_stations is not None:
                parts.append(f"Floor space allows at most {spec.max_stations} stations.")
    else:
        if "max_stations" in hidden:
            parts.append(
                rng.choice(
                    [
                        "We are re-using the existing line layout. Please balance the operations to get the best possible cycle time.",
                        "The line is being rebuilt on the current floor space. Please balance it for the fastest cycle time.",
                    ]
                )
            )
        else:
            parts.append(
                rng.choice(
                    [
                        f"The line has {spec.max_stations} stations. Distribute the operations over these stations so that the cycle time (the largest station load) is as small as possible.",
                        f"Use {spec.max_stations} stations and minimise the cycle time, i.e. the largest station load.",
                    ]
                )
            )
            if spec.cycle_time is not None:
                parts.append(f"The cycle time must not exceed {spec.cycle_time} s.")
    cons = describe_constraints(constraints or {})
    if cons:
        parts.append("Engineering notes: " + "; ".join(cons) + ".")
    parts.append("Operation times and precedence relations are in the line model.")
    return " ".join(parts)
