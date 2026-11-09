"""MCP server exposing one line-balancing episode (a task plus its environment state).

Run over stdio:
    uv run python -m linebal.server --tasks tasks/mvp.jsonl --task-id T1-001 [--fault-rate 0.1]

Constraint semantics are written into the tool descriptions and violations come back as structured
errors, following da Silva et al. (2025). Tools whose name starts with `admin_` are for the
experiment harness only and are never shown to an agent.
"""

from __future__ import annotations

import argparse
import json
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from .env import LineEnv, ToolFailure
from .faults import FaultConfig
from .tasks import load_tasks

SERVER_INSTRUCTIONS = (
    "Line-balancing workbench. Operations have times in seconds and precedence relations. A balance "
    "assigns every operation to a numbered station. Hard constraints: no station load above the cycle "
    "time limit; an operation never upstream of its predecessors; stations used within the station "
    "limit; linked operations share a station; incompatible operations do not; excluded stations are "
    "not used for that operation."
)


class Assignment(BaseModel):
    operation: int = Field(description="Operation id")
    station: int = Field(description="Station number, 1..n_stations")


def build_server(env: LineEnv) -> MCPServer:
    srv = MCPServer("linebal", instructions=SERVER_INSTRUCTIONS, version="0.1.0", log_level="ERROR")

    def run(name: str, **args) -> str:
        # Compact JSON keeps tool results (most of an agent's input tokens) small.
        try:
            return json.dumps(env.call(name, args), separators=(",", ":"))
        except ToolFailure as e:
            raise ToolError(json.dumps(e.to_dict(), separators=(",", ":"))) from e

    @srv.tool(description="Overview of the line: number of operations, total work content, the objective and limits from the work order (cycle time limit, station limit), constraint counts and current progress.")
    def get_line_summary() -> str:
        return run("get_line_summary")

    @srv.tool(description="List every operation with its time in seconds and its direct predecessors. An operation must be on the same station as, or a later station than, each of its predecessors.")
    def get_operations() -> str:
        return run("get_operations")

    @srv.tool(description="Hard constraints of the work order: cycle time limit (maximum load per station), station limit, linked pairs (must share a station), incompatible pairs (must not share a station) and stations excluded for specific operations.")
    def get_constraints() -> str:
        return run("get_constraints")

    @srv.tool(description="Current draft balance: the operations and load of each station, plus operations not yet assigned.")
    def get_balance() -> str:
        return run("get_balance")

    @srv.tool(description="Set the number of stations (numbered 1..n_stations). Reducing it unassigns the operations on removed stations.")
    def set_stations(n_stations: int) -> str:
        return run("set_stations", n_stations=n_stations)

    @srv.tool(description="Assign one or more operations to stations in one call, e.g. [{\"operation\": 3, \"station\": 1}]. Assigning an operation again moves it. Only malformed input is rejected; engineering violations are allowed in a draft, so use validate_balance to find them.")
    def assign_operations(assignments: list[Assignment]) -> str:
        return run("assign_operations", assignments=[a.model_dump() for a in assignments])

    @srv.tool(description="Check the current draft against every hard constraint and list each violation with the operations, stations and amounts involved.")
    def validate_balance() -> str:
        return run("validate_balance")

    @srv.tool(description="Key figures of the current draft: stations used, realised cycle time (largest station load), line efficiency, idle time and station loads.")
    def compute_kpis() -> str:
        return run("compute_kpis")

    @srv.tool(description="Submit the current balance as the final answer. Use status 'complete' when the balance meets the work order, or 'infeasible' when the requirements cannot all be met (explain the conflict). Report the number of stations and the cycle time of the submitted balance.")
    def submit_balance(
        status: Literal["complete", "infeasible"],
        reported_n_stations: int | None = None,
        reported_cycle_time: int | None = None,
        explanation: str = "",
    ) -> str:
        return run(
            "submit_balance",
            status=status,
            reported_n_stations=reported_n_stations,
            reported_cycle_time=reported_cycle_time,
            explanation=explanation,
        )

    @srv.tool(description="Send a question or an escalation to the production engineer who issued the work order.")
    def ask_engineer(message: str) -> str:
        return run("ask_engineer", message=message)

    @srv.tool(description="Harness only: export the full episode state and call log.")
    def admin_export() -> dict:
        return env.export()

    @srv.tool(description="Harness only: run a tool without fault injection and without counting it as an agent call.")
    def admin_call(tool: str, arguments: dict | None = None) -> dict:
        try:
            return env.call(tool, arguments or {}, source="harness")
        except ToolFailure as e:
            return e.to_dict()

    return srv


def main() -> None:
    ap = argparse.ArgumentParser(description="Line-balancing MCP server (stdio)")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--fault-rate", type=float, default=0.0)
    ap.add_argument("--fault-kinds", default="transient")
    ap.add_argument("--fault-seed", type=int, default=0)
    a = ap.parse_args()
    task = {t.task_id: t for t in load_tasks(a.tasks)}[a.task_id]
    faults = FaultConfig(rate=a.fault_rate, kinds=a.fault_kinds.split(","), seed=a.fault_seed)
    build_server(LineEnv(task, faults=faults)).run("stdio")


if __name__ == "__main__":
    main()
