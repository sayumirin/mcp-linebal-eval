"""Read and write line-balancing instances.

Two formats are supported:

* `.ALB` (Scholl, Becker, Boysen, Fliedner 2007): tagged blocks such as `<number of tasks>`,
  `<cycle time>`, `<task times>`, `<precedence relations>`, `<linked tasks>`,
  `<incompatible tasks>`, `<tasks excluded from station>`, `<end>`. Older files may separate the
  task id and time in `<task times>` with a space instead of a colon; both are accepted.
* `.IN2` (Scholl 1993 benchmark files): line 1 is n, the next n lines are task times, then
  `i,j` precedence pairs, optionally terminated by `-1,-1`.

The ALB spec asks for topologically numbered tasks; we accept any acyclic numbering because
relabelled instances (used to check for training-data contamination) break that order on purpose.
"""

from __future__ import annotations

import re
from pathlib import Path

from .model import Instance

_TAG = re.compile(r"^<([^>]+)>")


def _split_blocks(text: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _TAG.match(line)
        if m:
            current = m.group(1).strip().lower()
            if current == "end":
                break
            blocks.setdefault(current, [])
            continue
        if current is not None:
            blocks[current].append(line)
    return blocks


def _pair(line: str) -> tuple[int, int]:
    a, b = re.split(r"\s*,\s*", line.strip())[:2]
    return int(a), int(b)


def parse_alb(text: str, name: str = "instance") -> Instance:
    blocks = _split_blocks(text)
    if "number of tasks" not in blocks or "task times" not in blocks:
        raise ValueError("ALB file must contain <number of tasks> and <task times>")
    n = int(blocks["number of tasks"][0])
    times: dict[int, int] = {}
    for line in blocks["task times"]:
        parts = re.split(r"\s*:\s*|\s+", line.strip())
        times[int(parts[0])] = int(parts[1])
    if sorted(times) != list(range(1, n + 1)):
        raise ValueError(f"task ids must be 1..{n}, got {sorted(times)[:5]}...")
    precedences = [_pair(x) for x in blocks.get("precedence relations", [])]
    linked = [_pair(x) for x in blocks.get("linked tasks", [])]
    incompatible = [_pair(x) for x in blocks.get("incompatible tasks", [])]
    excluded: dict[int, list[int]] = {}
    for line in blocks.get("tasks excluded from station", []):
        task, stations = line.split(":", 1)
        excluded[int(task)] = [int(s) for s in re.split(r"\s*,\s*", stations.strip()) if s]
    cycle = int(blocks["cycle time"][0]) if blocks.get("cycle time") else None
    stations = int(blocks["number of stations"][0]) if blocks.get("number of stations") else None
    inst = Instance(
        name=name,
        times=times,
        precedences=precedences,
        linked=linked,
        incompatible=incompatible,
        excluded=excluded,
        cycle_time=cycle,
        n_stations=stations,
    )
    validate_instance(inst)
    return inst


def parse_in2(text: str, name: str = "instance") -> Instance:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    n = int(lines[0])
    times = {i + 1: int(lines[1 + i]) for i in range(n)}
    precedences = []
    for line in lines[1 + n :]:
        a, b = _pair(line)
        if a == -1 and b == -1:
            break
        precedences.append((a, b))
    inst = Instance(name=name, times=times, precedences=precedences)
    validate_instance(inst)
    return inst


def read_instance(path: str | Path, name: str | None = None) -> Instance:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    nm = name or path.stem
    if path.suffix.lower() == ".in2":
        return parse_in2(text, nm)
    return parse_alb(text, nm)


def write_alb(inst: Instance) -> str:
    out: list[str] = ["<number of tasks>", str(inst.n), ""]
    if inst.cycle_time is not None:
        out += ["<cycle time>", str(inst.cycle_time), ""]
    if inst.n_stations is not None:
        out += ["<number of stations>", str(inst.n_stations), ""]
    # Otto et al. (2013) files write the order strength with a decimal comma; we follow that.
    out += ["<order strength>", f"{100 * inst.order_strength():.3f}".replace(".", ","), ""]
    out.append("<task times>")
    out += [f"{i}:{inst.times[i]}" for i in inst.ops]
    out.append("")
    out.append("<precedence relations>")
    out += [f"{a},{b}" for a, b in sorted(inst.precedences)]
    out.append("")
    if inst.linked:
        out.append("<linked tasks>")
        out += [f"{a},{b}" for a, b in inst.linked]
        out.append("")
    if inst.incompatible:
        out.append("<incompatible tasks>")
        out += [f"{a},{b}" for a, b in inst.incompatible]
        out.append("")
    if inst.excluded:
        out.append("<tasks excluded from station>")
        out += [f"{i}:{','.join(str(s) for s in st)}" for i, st in sorted(inst.excluded.items())]
        out.append("")
    out.append("<end>")
    return "\n".join(out) + "\n"


def validate_instance(inst: Instance) -> None:
    ops = set(inst.ops)
    for a, b in inst.precedences + inst.linked + inst.incompatible:
        if a not in ops or b not in ops:
            raise ValueError(f"{inst.name}: relation ({a},{b}) refers to an unknown task")
    for i in inst.excluded:
        if i not in ops:
            raise ValueError(f"{inst.name}: excluded task {i} unknown")
    if any(t <= 0 for t in inst.times.values()):
        raise ValueError(f"{inst.name}: task times must be positive")
    inst.topological_order()  # raises on cycles
