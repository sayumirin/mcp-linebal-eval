"""A deterministic scripted "model" for tests and CI.

It reads the conversation, rebuilds the line from tool results and plans with the RPW heuristic.
The `sloppy` profile makes seeded, human-like mistakes (ignores engineering constraints, slips an
operation to the wrong station, skips validation, misreports KPIs, guesses missing parameters), so
every scoring path and the verifier loop get exercised. `perfect` never makes them.

The policy is stateless: each call re-derives what to do from the message history, and every
random decision is a fixed coin per (seed, decision name), so a run is reproducible.
"""

from __future__ import annotations

import json
import math
import random
import re
from typing import Any

from ..checker import kpis
from ..model import Balance, Instance, ProblemSpec
from ..solver import heuristic, rpw
from .llm import LLM, LLMResponse, ToolCall

PROFILES = {
    "perfect": dict(ignore_constraints=0.0, slip=0.0, validate=1.0, repair=1.0, misreport=0.0, ask=1.0, detect=1.0),
    "sloppy": dict(ignore_constraints=0.5, slip=0.25, validate=0.5, repair=0.7, misreport=0.15, ask=0.5, detect=0.5),
}
RETRY_CODES = ("temporarily_unavailable", "timeout", "locked")


class FakeLLM(LLM):
    def __init__(self, seed: int = 0, profile: str = "sloppy"):
        self.seed = seed
        self.profile = profile
        self.p = PROFILES[profile]
        self.name = f"fake-{profile}"
        self._n = 0

    def describe(self) -> dict:
        return {"name": self.name, "seed": self.seed, "profile": self.profile}

    def for_run(self, run_seed: int) -> FakeLLM:
        return FakeLLM(seed=run_seed, profile=self.profile)

    def coin(self, name: str, p: float) -> bool:
        return random.Random(f"{self.seed}:{name}").random() < p

    # ------------------------------------------------------------------ entry point
    async def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096) -> LLMResponse:
        if tools:
            content, calls = self._tool_step(messages)
        else:
            content, calls = self._direct(messages), []
        text_in = sum(len(json.dumps(m)) for m in messages) + (len(json.dumps(tools)) if tools else 0)
        out = content + json.dumps([c.arguments for c in calls])
        return LLMResponse(
            content=content,
            tool_calls=calls,
            usage={"input_tokens": text_in // 4, "output_tokens": max(1, len(out) // 4)},
            model=self.name,
            finish_reason="tool_calls" if calls else "stop",
        )

    def _call(self, name: str, args: dict) -> ToolCall:
        self._n += 1
        return ToolCall(id=f"call_{self._n}", name=name, arguments=args, raw_arguments=json.dumps(args))

    # ------------------------------------------------------------------ planning
    def _spec_from(self, summary: dict, engineer_answers: list[str], tag: str) -> tuple[ProblemSpec | None, bool]:
        """Return (spec, guessed). spec None means: ask first."""
        c, m = summary.get("cycle_time_limit_s"), summary.get("max_stations")
        obj = summary.get("objective", "")
        if c is not None or m is not None:
            problem = "SALBP-2" if "cycle time on" in obj else "SALBP-1"
            return ProblemSpec(problem, cycle_time=c, max_stations=m), False
        for ans in engineer_answers:
            mc = re.search(r"cycle time is (\d+)", ans)
            ms = re.search(r"have (\d+) stations", ans)
            if mc:
                return ProblemSpec("SALBP-1", cycle_time=int(mc.group(1))), False
            if ms:
                return ProblemSpec("SALBP-2", max_stations=int(ms.group(1))), False
        if engineer_answers or not self.coin(f"ask:{tag}", self.p["ask"]):
            total, longest = summary["total_work_content_s"], summary["longest_operation_s"]
            return ProblemSpec("SALBP-1", cycle_time=max(longest, math.ceil(total / 5))), True
        return None, False

    def _plan(self, inst: Instance, spec: ProblemSpec, careful: bool, tag: str) -> tuple[Balance | None, str]:
        """Returns (balance, reason). balance None means the agent concludes the order is infeasible."""
        if spec.cycle_time is not None and spec.cycle_time < inst.max_time:
            return None, f"Operation times up to {inst.max_time} s exceed the cycle time {spec.cycle_time} s."
        ignore = not careful and self.coin(f"ignore:{tag}", self.p["ignore_constraints"])
        work = inst if not ignore else Instance(inst.name, inst.times, inst.precedences)
        if spec.problem == "SALBP-1":
            bal = rpw(work, spec.cycle_time)
        else:
            res = heuristic(work, ProblemSpec("SALBP-2", spec.cycle_time, spec.max_stations))
            bal = res.balance
        if bal is None:
            return None, "I could not find any assignment that satisfies all requirements."
        if spec.max_stations is not None and kpis(inst, bal).n_stations > spec.max_stations:
            if careful or self.coin(f"detect:{tag}", self.p["detect"]):
                return None, f"The cycle time needs more than {spec.max_stations} stations."
        if not careful and self.coin(f"slip:{tag}", self.p["slip"]) and bal.n_stations > 1:
            rng = random.Random(f"{self.seed}:slipop:{tag}")
            op = rng.choice(sorted(bal.assignment))
            st = bal.assignment[op]
            bal.assignment[op] = st + 1 if st < bal.n_stations else st - 1
        return bal, ""

    def _report(self, inst: Instance, bal: Balance, tag: str) -> tuple[int, int]:
        k = kpis(inst, bal)
        n, ct = k.n_stations, k.cycle_time
        if self.coin(f"misreport:{tag}", self.p["misreport"]):
            n, ct = n - 1 if n > 1 else n + 1, ct - max(1, ct // 20)
        return n, ct

    # ------------------------------------------------------------------ tool mode
    def _tool_step(self, messages: list[dict]) -> tuple[str, list[ToolCall]]:
        calls_by_id: dict[str, tuple[str, dict]] = {}
        events: list[tuple[str, dict, str]] = []  # (tool, args, result text)
        for msg in messages:
            if msg["role"] == "assistant":
                for tc in msg.get("tool_calls") or []:
                    calls_by_id[tc["id"]] = (tc["function"]["name"], json.loads(tc["function"]["arguments"] or "{}"))
            elif msg["role"] == "tool":
                name, args = calls_by_id.get(msg["tool_call_id"], ("?", {}))
                events.append((name, args, msg["content"]))

        # retry a call that failed for a transient reason
        if events and any(code in events[-1][2] for code in RETRY_CODES):
            name, args, _ = events[-1]
            return f"The {name} call failed temporarily, retrying.", [self._call(name, args)]

        rejections = [k for k, e in enumerate(events) if e[0] == "submit_balance" and "rejected" in e[2]]
        since = rejections[-1] + 1 if rejections else 0
        careful = bool(rejections) and self.coin(f"repair:{len(rejections)}", self.p["repair"])
        tag = f"r{len(rejections)}"

        def latest(name: str) -> dict | None:
            for n, _, text in reversed(events):
                if n == name and not text.startswith("Error"):
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return None
            return None

        summary = latest("get_line_summary")
        if summary is None:
            return "First I will look at the line summary.", [self._call("get_line_summary", {})]
        answers = [json.loads(t).get("answer", "") for n, _, t in events if n == "ask_engineer" and not t.startswith("Error")]
        spec, guessed = self._spec_from(summary, answers, "spec")
        if spec is None:
            return "The work order has no cycle time or station count, so I will ask.", [
                self._call("ask_engineer", {"message": "The work order gives no cycle time or number of stations. What should I use?"})
            ]
        ops = latest("get_operations")
        if ops is None:
            return "Next I need the operations.", [self._call("get_operations", {})]
        cons = latest("get_constraints")
        if cons is None:
            return "And the constraints.", [self._call("get_constraints", {})]
        inst = Instance(
            name=summary["line"],
            times={o["id"]: o["time_s"] for o in ops["operations"]},
            precedences=[(p, o["id"]) for o in ops["operations"] for p in o["predecessors"]],
            linked=[tuple(x) for x in cons["linked"]],
            incompatible=[tuple(x) for x in cons["incompatible"]],
            excluded={e["operation"]: e["stations"] for e in cons["excluded"]},
        )
        after = events[since:]
        assigned = [e for e in after if e[0] == "assign_operations" and not e[2].startswith("Error")]
        bal, reason = self._plan(inst, spec, careful, tag)
        if bal is None:
            return reason, [self._call("submit_balance", {"status": "infeasible", "explanation": reason})]
        if not assigned:
            plan = [{"operation": op, "station": st} for op, st in sorted(bal.assignment.items())]
            note = " I am assuming a cycle time because none was given." if guessed else ""
            return f"I will set {bal.n_stations} stations and assign all operations.{note}", [
                self._call("set_stations", {"n_stations": bal.n_stations}),
                self._call("assign_operations", {"assignments": plan}),
            ]
        validated = [e for e in after if e[0] == "validate_balance" and not e[2].startswith("Error")]
        if not validated and self.coin(f"validate:{tag}", self.p["validate"]):
            return "Let me validate the draft.", [self._call("validate_balance", {})]
        if validated and '"feasible":false' in validated[-1][2] and not careful and self.coin(f"fix:{tag}", self.p["repair"]):
            good, _ = self._plan(inst, spec, True, tag + "fix")
            if good is not None and not any(e[0] == "assign_operations" for e in after[after.index(validated[-1]) :]):
                plan = [{"operation": op, "station": st} for op, st in sorted(good.assignment.items())]
                return "The validator found problems, so I re-plan.", [
                    self._call("set_stations", {"n_stations": good.n_stations}),
                    self._call("assign_operations", {"assignments": plan}),
                ]
        current = latest("get_balance")
        state_bal = bal
        if current is None:
            # the plan we sent is the state unless a later fix replaced it
            last_assign = json.loads(json.dumps([e for e in after if e[0] == "assign_operations"][-1][1]))
            last_set = [e for e in after if e[0] == "set_stations"]
            n_st = last_set[-1][1]["n_stations"] if last_set else bal.n_stations
            state_bal = Balance(n_st, {a["operation"]: a["station"] for a in last_assign["assignments"]})
        n, ct = self._report(inst, state_bal, tag)
        return "The balance is ready; submitting.", [
            self._call(
                "submit_balance",
                {"status": "complete", "reported_n_stations": n, "reported_cycle_time": ct, "explanation": "Balanced with the RPW rule."},
            )
        ]

    # ------------------------------------------------------------------ direct mode
    def _direct(self, messages: list[dict]) -> str:
        user = messages[-1]["content"]
        data = json.loads(user.split("Line model data (JSON):\n", 1)[1])
        summary, cons = data["line_summary"], data["constraints"]
        spec, guessed = self._spec_from(summary, [], "direct")
        if spec is None:
            return json.dumps({"status": "needs_clarification", "question": "What cycle time or number of stations should I use?"})
        inst = Instance(
            name=summary["line"],
            times={o["id"]: o["time_s"] for o in data["operations"]},
            precedences=[(p, o["id"]) for o in data["operations"] for p in o["predecessors"]],
            linked=[tuple(x) for x in cons["linked"]],
            incompatible=[tuple(x) for x in cons["incompatible"]],
            excluded={e["operation"]: e["stations"] for e in cons["excluded"]},
        )
        bal, reason = self._plan(inst, spec, False, "direct")
        if bal is None:
            return json.dumps({"status": "infeasible", "explanation": reason})
        n, ct = self._report(inst, bal, "direct")
        out: dict[str, Any] = {
            "status": "complete",
            "n_stations": bal.n_stations,
            "assignments": {str(k): v for k, v in sorted(bal.assignment.items())},
            "reported_n_stations": n,
            "reported_cycle_time": ct,
            "explanation": "Balanced with the RPW rule." + (" Cycle time assumed." if guessed else ""),
        }
        return "```json\n" + json.dumps(out) + "\n```"
