"""A2: ReAct plus an external verifier in the loop (LLM-Modulo, Kambhampati et al. 2024; Gundawar
et al. 2024). The agent has exactly the A1 tools. When it submits a balance with status
'complete', the harness checks it with the same checker as validate_balance. A rejected submission
never reaches the environment; the violation list goes back to the agent instead. After
`max_rounds` rejections the episode is handed to the engineer, in the spirit of the retry limit
with a safe fallback in Vyas and Mercangoz (2024).

The verifier only checks what the work order states (hidden T5 parameters are not known to it), and
it cannot check an 'infeasible' claim, so such submissions pass straight through."""

from __future__ import annotations

import json

from .base import McpSession, RunConfig
from .llm import LLM, ToolCall
from .prompts import VERIFIER_ESCALATION, verifier_feedback
from .react import run_tool_loop


async def run_modulo(session: McpSession, llm: LLM, cfg: RunConfig) -> dict:
    rounds = 0
    events: list[dict] = []

    async def intercept(tc: ToolCall):
        nonlocal rounds
        if tc.name != "submit_balance" or not isinstance(tc.arguments, dict) or tc.arguments.get("status") != "complete":
            return None
        verdict = await session.admin("validate_balance")
        if verdict.get("feasible"):
            events.append({"event": "verifier_pass", "round": rounds + 1})
            return None
        rounds += 1
        balance = await session.admin("get_balance")
        events.append(
            {
                "event": "verifier_reject",
                "round": rounds,
                "n_violations": verdict.get("n_violations"),
                "types": sorted({v["type"] for v in verdict.get("violations", [])}),
                "attempt": {"arguments": tc.arguments, "balance": balance},
            }
        )
        if rounds >= cfg.max_rounds:
            events.append({"event": "verifier_escalation", "round": rounds})
            return json.dumps({"rejected": True, "message": VERIFIER_ESCALATION.format(max_rounds=cfg.max_rounds)}), True, True
        text = verifier_feedback(verdict.get("violations", []), rounds, cfg.max_rounds)
        return json.dumps({"rejected": True, "message": text}), True, False

    out = await run_tool_loop(session, llm, cfg, intercept)
    out["events"] = events + out["events"]
    if any(e["event"] == "verifier_escalation" for e in events):
        out["end"] = "escalated"
    out["verifier_rounds"] = rounds
    return out
