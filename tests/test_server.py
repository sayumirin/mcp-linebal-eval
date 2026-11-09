import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

from conftest import make_task
from linebal.env import PUBLIC_TOOLS, LineEnv, ToolFailure
from linebal.faults import FaultConfig
from linebal.server import build_server
from linebal.tasks import save_tasks


def call(client, name, args=None):
    async def go():
        r = await client.call_tool(name, args or {})
        return r.content[0].text, r.is_error

    return go()


def test_tool_listing_and_draft_semantics(valve_task):
    async def main():
        env = LineEnv(valve_task)
        async with Client(build_server(env)) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert set(PUBLIC_TOOLS) <= names
            text, err = await call(c, "assign_operations", {"assignments": [{"operation": 1, "station": 1}]})
            assert err and "no_stations" in text
            await call(c, "set_stations", {"n_stations": 2})
            text, err = await call(c, "assign_operations", {"assignments": [{"operation": 1, "station": 1}, {"operation": 99, "station": 1}]})
            assert err and "operation 99 does not exist" in text
            assert env.assignment == {}  # atomic: nothing applied
            # an engineering violation is allowed in a draft
            text, err = await call(c, "assign_operations", {"assignments": [{"operation": 2, "station": 1}, {"operation": 1, "station": 2}]})
            assert not err
            v = json.loads((await call(c, "validate_balance"))[0])
            assert not v["feasible"]
            assert any(x["type"] == "precedence" for x in v["violations"])

    asyncio.run(main())


def test_engineer_reveals_hidden_parameter_once():
    task = make_task(type="T5", hidden={"max_stations": 5}, request="Please balance the line.")
    env = LineEnv(task)
    assert env.get_line_summary()["max_stations"] is None
    ans = env.call("ask_engineer", {"message": "How many stations?"})["answer"]
    assert "5 stations" in ans
    assert env.get_line_summary()["max_stations"] == 5
    again = env.call("ask_engineer", {"message": "Thanks, anything else?"})["answer"]
    assert "proceed" in again.lower()


def test_engineer_reply_does_not_depend_on_feasibility(valve_task):
    infeasible = make_task(type="T4", expected="infeasible", spec={"problem": "SALBP-1", "cycle_time": 100, "max_stations": None})
    a = LineEnv(valve_task).call("ask_engineer", {"message": "This looks infeasible."})
    b = LineEnv(infeasible).call("ask_engineer", {"message": "This looks infeasible."})
    assert a == b


def test_fault_injection_is_seeded_and_skips_submit(valve_task):
    def pattern(seed):
        env = LineEnv(valve_task, faults=FaultConfig(rate=0.5, seed=seed))
        out = []
        for _ in range(20):
            try:
                env.call("get_line_summary")
                out.append(0)
            except ToolFailure as e:
                assert e.code == "temporarily_unavailable"
                out.append(1)
        env.call("submit_balance", {"status": "infeasible"})  # never faulted
        return out

    assert pattern(1) == pattern(1)
    assert pattern(1) != pattern(2)
    assert 0 < sum(pattern(1)) < 20


def test_stdio_transport(tmp_path: Path, valve_task):
    path = tmp_path / "tasks.jsonl"
    save_tasks([valve_task], path)

    async def main():
        params = StdioServerParameters(command=sys.executable, args=["-m", "linebal.server", "--tasks", str(path), "--task-id", valve_task.task_id])
        async with Client(params) as c:
            text, err = await call(c, "get_line_summary")
            assert not err and json.loads(text)["n_operations"] == 12

    asyncio.run(main())
