"""The line-balancing environment behind the MCP server: state, tool semantics and the call log.

Design choice (as in real engineering software): write tools only reject malformed input such as an
unknown operation id. They do not reject engineering violations, so a draft balance may break the
cycle time or a precedence relation. Whether an agent calls `validate_balance` on its own is part
of what the experiments observe.
"""

from __future__ import annotations

import time
from typing import Any

from .checker import check, kpis
from .engineer import Engineer
from .faults import FaultConfig, FaultInjector, READ_TOOLS
from .model import Balance, Instance, ProblemSpec
from .tasks import Task, build_instance

PUBLIC_TOOLS = (
    "get_line_summary",
    "get_operations",
    "get_constraints",
    "get_balance",
    "set_stations",
    "assign_operations",
    "validate_balance",
    "compute_kpis",
    "submit_balance",
    "ask_engineer",
)


class ToolFailure(Exception):
    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, **self.details}}


class LineEnv:
    def __init__(self, task: Task, instance: Instance | None = None, faults: FaultConfig | None = None):
        self.task = task
        self.inst = instance or build_instance(task)
        self.engineer = Engineer(task)
        self.faults = FaultInjector(faults)
        self.n_stations = 0
        self.assignment: dict[int, int] = {}
        self.submissions: list[dict] = []
        self.calls: list[dict] = []
        self.t0 = time.time()

    # ------------------------------------------------------------------ state
    @property
    def balance(self) -> Balance:
        return Balance(self.n_stations, dict(self.assignment))

    @property
    def visible_spec(self) -> ProblemSpec:
        return self.task.visible_spec(self.engineer.revealed)

    @property
    def submitted(self) -> bool:
        return bool(self.submissions)

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict | None = None, *, source: str = "agent") -> dict:
        """Run one tool call with fault injection and logging. Raises ToolFailure on errors."""
        args = args or {}
        entry: dict[str, Any] = {"t": round(time.time() - self.t0, 3), "tool": name, "args": args, "source": source}
        fault = self.faults.draw(name) if source == "agent" else None
        try:
            if name not in PUBLIC_TOOLS:
                raise ToolFailure("unknown_tool", f"Unknown tool {name!r}")
            if fault == "transient":
                raise ToolFailure("temporarily_unavailable", "The line model service is temporarily unavailable. Please retry.")
            if fault == "timeout":
                raise ToolFailure("timeout", "The request timed out after 30 s; nothing was changed.")
            if fault == "write_reject":
                raise ToolFailure("locked", "The line model is locked by another user; the change was not applied.")
            result = getattr(self, name)(**args)
            if fault == "partial":
                result = self._truncate(result)
            entry.update(ok=True, result=result)
            return result
        except ToolFailure as e:
            entry.update(ok=False, error=e.to_dict()["error"])
            raise
        except TypeError as e:  # wrong argument names from a confused agent
            err = ToolFailure("bad_arguments", str(e))
            entry.update(ok=False, error=err.to_dict()["error"])
            raise err from e
        finally:
            if fault:
                entry["fault"] = fault
            self.calls.append(entry)

    @staticmethod
    def _truncate(result: dict) -> dict:
        out = dict(result)
        for key in ("operations", "stations"):
            if isinstance(out.get(key), list) and len(out[key]) > 1:
                out[key] = out[key][: len(out[key]) // 2]
        return out

    # ------------------------------------------------------------------ read tools
    def get_line_summary(self) -> dict:
        spec = self.visible_spec
        if spec.problem == "SALBP-1" and spec.cycle_time is not None:
            objective = f"minimise the number of stations at cycle time {spec.cycle_time} s"
        elif spec.problem == "SALBP-2" and spec.max_stations is not None:
            objective = f"minimise the cycle time on {spec.max_stations} stations"
        else:
            objective = "not specified in the work order"
        known = spec.cycle_time is not None or spec.max_stations is not None
        return {
            "line": self.inst.name,
            "n_operations": self.inst.n,
            "total_work_content_s": self.inst.total_time,
            "longest_operation_s": self.inst.max_time,
            "objective": objective,
            "cycle_time_limit_s": spec.cycle_time if known else None,
            "max_stations": spec.max_stations if known else None,
            "n_precedence_relations": len(self.inst.precedences),
            "n_engineering_constraints": {
                "linked": len(self.inst.linked),
                "incompatible": len(self.inst.incompatible),
                "excluded": len(self.inst.excluded),
            },
            "stations_configured": self.n_stations,
            "operations_assigned": len(self.assignment),
        }

    def get_operations(self) -> dict:
        pred = self.inst.predecessors()
        return {
            "operations": [{"id": i, "time_s": self.inst.times[i], "predecessors": pred[i]} for i in self.inst.ops],
        }

    def get_constraints(self) -> dict:
        spec = self.visible_spec
        return {
            "cycle_time_limit_s": spec.cycle_time,
            "max_stations": spec.max_stations,
            "linked": [list(p) for p in self.inst.linked],
            "incompatible": [list(p) for p in self.inst.incompatible],
            "excluded": [{"operation": i, "stations": s} for i, s in sorted(self.inst.excluded.items())],
            "precedence": "every operation must be on the same or a later station than each of its predecessors (see get_operations)",
        }

    def get_balance(self) -> dict:
        bal = self.balance
        loads = bal.loads(self.inst)
        return {
            "n_stations": self.n_stations,
            "stations": [
                {"station": k, "operations": ops, "load_s": loads.get(k, 0)} for k, ops in bal.station_ops().items()
            ],
            "unassigned": [i for i in self.inst.ops if i not in self.assignment],
        }

    def validate_balance(self) -> dict:
        res = check(self.inst, self.balance, self.visible_spec)
        out = res.to_dict(limit=25)
        if self.visible_spec.cycle_time is None and self.visible_spec.max_stations is None:
            out.setdefault("warnings", []).append("No cycle time or station limit is set, so neither was checked.")
        return out

    def compute_kpis(self) -> dict:
        k = kpis(self.inst, self.balance).to_dict()
        k["note"] = "n_stations counts stations with at least one operation; cycle_time is the largest station load"
        return k

    # ------------------------------------------------------------------ write tools
    def set_stations(self, n_stations: int) -> dict:
        n = int(n_stations)
        if not 1 <= n <= self.inst.n:
            raise ToolFailure("invalid_value", f"n_stations must be between 1 and {self.inst.n}", value=n)
        dropped = sorted(op for op, st in self.assignment.items() if st > n)
        for op in dropped:
            del self.assignment[op]
        self.n_stations = n
        out: dict[str, Any] = {"n_stations": n}
        if dropped:
            out["unassigned_by_change"] = dropped
        return out

    def assign_operations(self, assignments: list[dict]) -> dict:
        if self.n_stations < 1:
            raise ToolFailure("no_stations", "No stations are configured yet; call set_stations first.")
        if not isinstance(assignments, list) or not assignments:
            raise ToolFailure("bad_arguments", "assignments must be a non-empty list of {operation, station}")
        errors, staged = [], {}
        for a in assignments:
            try:
                op, st = int(a["operation"]), int(a["station"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"malformed entry {a!r}")
                continue
            if op not in self.inst.times:
                errors.append(f"operation {op} does not exist")
            elif not 1 <= st <= self.n_stations:
                errors.append(f"station {st} does not exist (stations 1..{self.n_stations})")
            elif op in staged and staged[op] != st:
                errors.append(f"operation {op} assigned to two stations in one call")
            else:
                staged[op] = st
        if errors:
            raise ToolFailure("invalid_assignment", "No changes applied: " + "; ".join(errors[:10]), n_errors=len(errors))
        self.assignment.update(staged)
        loads = self.balance.loads(self.inst)
        return {
            "updated": sorted(staged),
            "station_loads_s": {str(k): v for k, v in sorted(loads.items())},
            "unassigned_count": sum(1 for i in self.inst.ops if i not in self.assignment),
        }

    def submit_balance(
        self,
        status: str,
        reported_n_stations: int | None = None,
        reported_cycle_time: int | None = None,
        explanation: str = "",
    ) -> dict:
        if status not in ("complete", "infeasible"):
            raise ToolFailure("invalid_value", "status must be 'complete' or 'infeasible'", value=status)
        self.submissions.append(
            {
                "status": status,
                "reported_n_stations": reported_n_stations,
                "reported_cycle_time": reported_cycle_time,
                "explanation": explanation,
                "balance": self.balance.to_dict(),
                "spec_visible": self.visible_spec.to_dict(),
                "t": round(time.time() - self.t0, 3),
            }
        )
        return {"received": True, "status": status}

    def ask_engineer(self, message: str) -> dict:
        reply = self.engineer.ask(message)
        return {"answer": reply.answer}

    # ------------------------------------------------------------------ export
    def export(self) -> dict:
        return {
            "task_id": self.task.task_id,
            "balance": self.balance.to_dict(),
            "submissions": self.submissions,
            "calls": self.calls,
            "engineer": self.engineer.log,
            "revealed": self.engineer.revealed,
            "faults_injected": self.faults.injected,
        }


def is_read_tool(name: str) -> bool:
    return name in READ_TOOLS
