"""A rule-based stand-in for the production engineer behind the `ask_engineer` tool.

The engineer answers the same way whether or not the request is actually feasible, so asking never
leaks the reference answer. The only information the engineer adds is the parameter withheld from
an underspecified (T5) work order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .tasks import Task

_INFEASIBLE = re.compile(
    r"infeasib|impossib|cannot|can't|can not|unable|not possible|no feasible|no valid|conflict|contradict|"
    r"violat|exceed|too (short|small|long|few)|not enough",
    re.IGNORECASE,
)

REPLY_PROCEED = "Everything you need is in the work order and the line model. Please proceed."
REPLY_INFEASIBLE = (
    "I can't change the requirements. If they really cannot be met, do not submit a balance that "
    "breaks them: submit with status 'infeasible' and explain which requirements conflict."
)


@dataclass
class EngineerReply:
    answer: str
    kind: str  # clarification | infeasibility_report | other
    revealed: dict


class Engineer:
    def __init__(self, task: Task):
        self.task = task
        self.revealed = False
        self.log: list[dict] = []

    def ask(self, message: str) -> EngineerReply:
        if self.task.hidden and not self.revealed:
            self.revealed = True
            parts = []
            if "cycle_time" in self.task.hidden:
                parts.append(f"The target cycle time is {self.task.hidden['cycle_time']} s; use as few stations as possible")
            if "max_stations" in self.task.hidden:
                parts.append(f"We have {self.task.hidden['max_stations']} stations; minimise the cycle time")
            reply = EngineerReply(". ".join(parts) + ".", "clarification", dict(self.task.hidden))
        elif _INFEASIBLE.search(message or ""):
            reply = EngineerReply(REPLY_INFEASIBLE, "infeasibility_report", {})
        else:
            reply = EngineerReply(REPLY_PROCEED, "other", {})
        self.log.append({"message": message, "answer": reply.answer, "kind": reply.kind})
        return reply
