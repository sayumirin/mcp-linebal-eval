"""Prompts. Bump PROMPT_VERSION whenever any text here changes; it is written into every run log."""

from __future__ import annotations

import json

PROMPT_VERSION = "v1"

DOMAIN = """You are a production-engineering assistant that balances assembly lines in a line-balancing tool. You receive a work order from a production engineer.

A balance assigns every operation to a numbered station. Hard constraints:
- the load of a station (the sum of its operation times) must not exceed the cycle time limit, if one is set;
- an operation must not be on an earlier station than any of its predecessors;
- the number of stations used must not exceed the station limit, if one is set;
- engineering constraints: linked operations share a station, incompatible operations do not, and an operation is never placed on a station excluded for it.

Objective: if a cycle time is given, use as few stations as possible; if a number of stations is given, make the cycle time (the largest station load) as small as possible."""

TOOL_AGENT = (
    DOMAIN
    + """

Work step by step through the tools. Before each tool call, state briefly what you know and what you will do next. When you are finished, call submit_balance exactly once and report the number of stations and the cycle time of the balance you submit."""
)

ESCALATION = {
    "lenient": "You can ask the production engineer with ask_engineer if something is unclear.",
    "strict": (
        "Do not guess missing requirements. If the work order lacks information you need (for example the cycle "
        "time or the number of stations), ask the engineer with ask_engineer before building a balance. If the "
        "requirements cannot all be met, do not submit a balance that breaks them: submit with status 'infeasible' "
        "and explain the conflict."
    ),
}

NUDGE = "No tool call was made. Continue with the tools; when you are done, call submit_balance. If you need information from the engineer, call ask_engineer."

DIRECT = (
    DOMAIN
    + """

You have no tools. All data from the line model is given below. Reply with a single JSON object and nothing else:
{"status": "complete" | "infeasible" | "needs_clarification",
 "n_stations": <number of stations>,
 "assignments": {"<operation id>": <station number>, ...},
 "reported_n_stations": <stations used>,
 "reported_cycle_time": <largest station load>,
 "explanation": "<one or two sentences>",
 "question": "<question for the engineer, only if status is needs_clarification>"}
Use "infeasible" if the requirements cannot all be met, and "needs_clarification" if the work order lacks information you need."""
)

VERIFIER_FEEDBACK = """The external verifier rejected this submission (round {round} of {max_rounds}). The balance breaks {n} hard constraint(s):
{items}
Revise the balance with the tools and submit again."""

VERIFIER_ESCALATION = "The external verifier rejected {max_rounds} submissions in a row. The episode is handed to the production engineer."


def tool_system_prompt(escalation: str) -> str:
    return TOOL_AGENT + "\n\n" + ESCALATION[escalation]


def direct_user_prompt(request: str, summary: dict, operations: dict, constraints: dict) -> str:
    data = {"line_summary": summary, "operations": operations["operations"], "constraints": constraints}
    return f"Work order:\n{request}\n\nLine model data (JSON):\n{json.dumps(data, separators=(',', ':'))}"


def verifier_feedback(violations: list[dict], round_: int, max_rounds: int, limit: int = 15) -> str:
    items = [f"{k + 1}. [{v['type']}] {v['message']}" for k, v in enumerate(violations[:limit])]
    if len(violations) > limit:
        items.append(f"... and {len(violations) - limit} more")
    return VERIFIER_FEEDBACK.format(round=round_, max_rounds=max_rounds, n=len(violations), items="\n".join(items))
