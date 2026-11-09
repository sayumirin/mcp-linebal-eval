"""Programmatic scoring of one run. No LLM judge: everything is recomputed from the final state.

Definitions
  feasible            submitted with status 'complete' and every hard constraint of the real work
                      order holds (for T5 that includes the parameter the engineer reveals)
  optimal             feasible and equal to the reference optimum (stations for SALBP-1, cycle
                      time for SALBP-2)
  near_optimal        feasible and within one station (SALBP-1) or 5% of the cycle time (SALBP-2)
  report_consistent   the reported stations and cycle time equal the recomputed ones
  silent_error        a 'complete' submission that is infeasible, or any 'complete' submission on
                      an infeasible (T4) task: the agent claims success but the balance is unusable
  success             T1-T3: feasible. T4: the conflict is reported (status 'infeasible', an
                      escalation to the engineer, or the A2 hand-over) and no balance is submitted.
                      T5: the agent asks before writing anything and then submits a feasible balance.

Failure labels follow tau-bench / MCP-Atlas style categories:
  wrong_parameter (cycle time or station limit broken), constraint_violation (precedence or an
  engineering constraint broken), partial (operations left unassigned), premature_stop,
  step_limit, report_mismatch, false_infeasible, assumed_parameter (T5 guessed), no_validation
  (failed without ever calling validate_balance), format_error, crash, output_limit (a reply was
  cut at max_tokens and nothing was submitted).
"""

from __future__ import annotations

from .checker import ENGINEERING_TYPES, check, kpis, objective_value
from .model import Balance, Instance
from .tasks import Task

WRITE_TOOLS = ("set_stations", "assign_operations")


def score_run(task: Task, inst: Instance, ref: dict | None, export: dict, agent: dict) -> dict:
    spec = task.problem_spec  # the real requirements, including hidden values
    subs = export.get("submissions", [])
    final = subs[-1] if subs else None
    status = final["status"] if final else None
    end = agent.get("end", "crash")
    agent_calls = [c for c in export.get("calls", []) if c.get("source") == "agent"]
    asks = [c for c in agent_calls if c["tool"] == "ask_engineer" and c.get("ok")]
    engineer = export.get("engineer", [])
    kinds = [e["kind"] for e in engineer]

    bal = Balance.from_dict(final["balance"]) if final else Balance.from_dict(export["balance"])
    res = check(inst, bal, spec)
    k = kpis(inst, bal)
    counts = res.counts()

    complete = status == "complete"
    feasible = complete and res.feasible and task.expected == "balance"
    obj = objective_value(spec, k) if feasible else None
    ref_obj = ref.get("objective") if ref else None
    gap = (obj - ref_obj) if (obj is not None and ref_obj is not None) else None
    gap_rel = (gap / ref_obj) if (gap is not None and ref_obj) else None
    optimal = bool(feasible and gap is not None and gap <= 0)
    if spec.problem == "SALBP-1":
        near = bool(feasible and gap is not None and gap <= 1)
    else:
        near = bool(feasible and gap_rel is not None and gap_rel <= 0.05)

    report_consistent = None
    if complete:
        report_consistent = final.get("reported_n_stations") == k.n_stations and final.get("reported_cycle_time") == k.cycle_time

    escalated_harness = end == "escalated"
    reported_conflict = status == "infeasible" or "infeasibility_report" in kinds or escalated_harness
    first_write = next((i for i, c in enumerate(agent_calls) if c["tool"] in WRITE_TOOLS), None)
    first_ask = next((i for i, c in enumerate(agent_calls) if c["tool"] == "ask_engineer"), None)
    asked_first = first_ask is not None and (first_write is None or first_ask < first_write)
    if end == "asked":  # A0 asking for clarification instead of answering
        asked_first = True

    if task.type in ("T1", "T2", "T3"):
        success = feasible
    elif task.type == "T4":
        success = (not complete) and reported_conflict
    else:  # T5
        success = asked_first and feasible and export.get("revealed", False)

    silent = bool(complete and (task.expected == "infeasible" or not res.feasible))
    false_infeasible = status == "infeasible" and task.expected == "balance"
    validated = any(c["tool"] == "validate_balance" for c in agent_calls)

    labels: list[str] = []
    if end == "crash":
        labels.append("crash")
    if end == "format_error":
        labels.append("format_error")
    truncated = any(e.get("event") == "truncated" for e in agent.get("events", [])) or any(
        c.get("finish_reason") == "length" for c in agent.get("llm_calls", [])
    )
    if final is None and end == "step_limit":
        labels.append("step_limit")
    if final is None and end in ("stopped", "format_error") and truncated:
        labels.insert(0, "output_limit")  # the reply hit max_tokens: a harness limit, not the model stopping
    elif final is None and end == "stopped":
        labels.append("premature_stop")
    if task.type == "T5" and complete and not asked_first:
        labels.append("assumed_parameter")
    if false_infeasible:
        labels.append("false_infeasible")
    if complete:
        if counts["unassigned"]:
            labels.append("partial")
        if counts["cycle_time"] or counts["station_limit"]:
            labels.append("wrong_parameter")
        if counts["precedence"] or any(counts[t] for t in ENGINEERING_TYPES) or counts["invalid_station"]:
            labels.append("constraint_violation")
        if report_consistent is False:
            labels.append("report_mismatch")
    if not success and not validated and task.type != "T4" and agent.get("arch") != "A0":
        labels.append("no_validation")
    primary = labels[0] if (labels and not success) else ("ok" if success else "other")

    usage_in = sum(c.get("usage", {}).get("input_tokens", 0) for c in agent.get("llm_calls", []))
    usage_out = sum(c.get("usage", {}).get("output_tokens", 0) for c in agent.get("llm_calls", []))
    return {
        "success": bool(success),
        "status": status,
        "end": end,
        "feasible": bool(feasible),
        "optimal": optimal,
        "near_optimal": near,
        "objective": obj,
        "reference": ref_obj,
        "reference_proven": (ref or {}).get("status") == "optimal",
        "gap": gap,
        "gap_rel": round(gap_rel, 4) if gap_rel is not None else None,
        "report_consistent": report_consistent,
        "silent_error": silent,
        "false_infeasible": bool(false_infeasible),
        "reported_conflict": bool(reported_conflict),
        "asked": len(asks),
        "asked_first": bool(asked_first),
        "engineer_kinds": kinds,
        "escalated_harness": escalated_harness,
        "violations": {t: n for t, n in counts.items() if n},
        "kpis": {"n_stations": k.n_stations, "cycle_time": k.cycle_time, "line_efficiency": round(k.line_efficiency, 4)},
        "validated": validated,
        "verifier_rounds": agent.get("verifier_rounds", 0),
        "labels": labels,
        "primary_failure": primary,
        "cost": {
            "llm_calls": len(agent.get("llm_calls", [])),
            "input_tokens": usage_in,
            "output_tokens": usage_out,
            "tool_calls": len(agent_calls),
            "tool_errors": sum(1 for c in agent_calls if not c.get("ok")),
            "faults_injected": len(export.get("faults_injected", [])),
        },
    }
