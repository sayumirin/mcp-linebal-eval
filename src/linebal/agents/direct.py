"""A0: direct prompting. All line data goes into one prompt and the model answers once with a JSON
balance, without tools. The harness then writes that balance into the environment through the same
MCP server, so it is scored exactly like the tool-using agents. Lower-bound baseline
(Gundawar et al. 2024 use the same kind of baseline)."""

from __future__ import annotations

from .base import McpSession, RunConfig, parse_json_answer
from .llm import LLM
from .prompts import DIRECT, direct_user_prompt


async def run_direct(session: McpSession, llm: LLM, cfg: RunConfig) -> dict:
    task = session.task
    summary = await session.admin("get_line_summary")
    ops = await session.admin("get_operations")
    cons = await session.admin("get_constraints")
    messages = [{"role": "system", "content": DIRECT}, {"role": "user", "content": direct_user_prompt(task.request, summary, ops, cons)}]
    resp = await llm.chat(messages, tools=None, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    events: list[dict] = []
    llm_log = [resp.to_log()]
    answer = parse_json_answer(resp.content)
    if answer is None:
        events.append({"event": "format_error", "detail": "no JSON object in reply"})
        return {"llm_calls": llm_log, "events": events, "end": "format_error"}

    status = answer.get("status")
    if status == "needs_clarification":
        events.append({"event": "clarification_request", "question": answer.get("question", "")})
        return {"llm_calls": llm_log, "events": events, "end": "asked"}
    if status not in ("complete", "infeasible"):
        events.append({"event": "format_error", "detail": f"status {status!r}"})
        return {"llm_calls": llm_log, "events": events, "end": "format_error"}

    if status == "complete":
        raw = answer.get("assignments") or {}
        assignments, bad = [], 0
        try:
            pairs = [(int(k), int(v)) for k, v in raw.items()]
        except (AttributeError, TypeError, ValueError):
            pairs, bad = [], len(raw) if hasattr(raw, "__len__") else 1
        stations = answer.get("n_stations")
        try:
            n_st = int(stations) if stations is not None else max((v for _, v in pairs), default=1)
        except (TypeError, ValueError):
            n_st = max((v for _, v in pairs), default=1)
        n_st = max(1, min(n_st, summary["n_operations"]))
        valid_ops = {o["id"] for o in ops["operations"]}
        for op, st in pairs:
            if op in valid_ops and 1 <= st <= n_st:
                assignments.append({"operation": op, "station": st})
            else:
                bad += 1
        await session.admin("set_stations", {"n_stations": n_st})
        if assignments:
            await session.admin("assign_operations", {"assignments": assignments})
        if bad:
            events.append({"event": "invalid_entries", "count": bad})

    def as_int(x):
        try:
            return int(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    await session.admin(
        "submit_balance",
        {
            "status": status,
            "reported_n_stations": as_int(answer.get("reported_n_stations")),
            "reported_cycle_time": as_int(answer.get("reported_cycle_time")),
            "explanation": str(answer.get("explanation", ""))[:2000],
        },
    )
    return {"llm_calls": llm_log, "events": events, "end": "submitted"}
