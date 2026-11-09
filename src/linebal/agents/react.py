"""A1: ReAct over MCP tools. The model reasons briefly, calls a tool, reads the result and repeats
until it calls submit_balance. Validation tools are available, but whether to use them is the
model's choice."""

from __future__ import annotations

from typing import Awaitable, Callable

from .base import McpSession, RunConfig
from .llm import LLM, ToolCall
from .prompts import NUDGE, tool_system_prompt

# A hook that may intercept a tool call before it reaches the server. It returns None to let the
# call through, or (text, is_error, end_episode) to answer it on the server's behalf.
Interceptor = Callable[[ToolCall], Awaitable["tuple[str, bool, bool] | None"]]


async def run_tool_loop(session: McpSession, llm: LLM, cfg: RunConfig, intercept: Interceptor | None = None) -> dict:
    messages: list[dict] = [
        {"role": "system", "content": tool_system_prompt(cfg.escalation)},
        {"role": "user", "content": session.task.request},
    ]
    tools = session.openai_tools()
    llm_log: list[dict] = []
    events: list[dict] = []
    nudged = False
    end = "step_limit"
    for step in range(cfg.max_steps):
        resp = await llm.chat(messages, tools=tools, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
        llm_log.append(resp.to_log())
        messages.append(resp.assistant_message(include_reasoning=llm.send_reasoning))
        if resp.finish_reason == "length":
            events.append({"event": "truncated", "step": step})
        if not resp.tool_calls:
            if nudged:
                end = "stopped"
                break
            nudged = True
            events.append({"event": "nudge", "step": step})
            messages.append({"role": "user", "content": NUDGE})
            continue
        finished = False
        for tc in resp.tool_calls:
            handled = await intercept(tc) if intercept is not None else None
            if handled is not None:
                text, is_error, stop = handled
            else:
                text, is_error = await session.call(tc.name, tc.arguments)
                stop = tc.name == "submit_balance" and not is_error
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
            if stop:
                finished = True
                break  # anything after a final submit in the same turn is ignored
        if finished:
            end = "submitted"  # A2 relabels this as "escalated" when its verifier gave up
            break
    return {"llm_calls": llm_log, "events": events, "end": end}


async def run_react(session: McpSession, llm: LLM, cfg: RunConfig) -> dict:
    return await run_tool_loop(session, llm, cfg)
