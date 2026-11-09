from conftest import make_task
from linebal.env import LineEnv
from linebal.scoring import score_run
from linebal.tasks import build_instance

OPT = {8: 1, 9: 1, 1: 2, 10: 2, 11: 3, 12: 3, 3: 4, 4: 4, 5: 4, 2: 5, 6: 5, 7: 5}
REF = {"status": "optimal", "objective": 240}


def run(task, steps, agent_end="submitted", arch="A1"):
    env = LineEnv(task)
    for name, args in steps:
        env.call(name, args)
    return score_run(task, build_instance(task), REF, env.export(), {"end": agent_end, "llm_calls": [], "arch": arch})


def assign(mapping):
    return ("assign_operations", {"assignments": [{"operation": o, "station": s} for o, s in mapping.items()]})


def test_optimal_and_consistent(valve_task):
    s = run(valve_task, [("set_stations", {"n_stations": 5}), assign(OPT), ("submit_balance", {"status": "complete", "reported_n_stations": 5, "reported_cycle_time": 240})])
    assert s["success"] and s["feasible"] and s["optimal"] and s["report_consistent"]
    assert not s["silent_error"]
    assert s["primary_failure"] == "ok"


def test_suboptimal_misreported(valve_task):
    sub = dict(OPT)
    sub[2] = 4  # station 4 becomes 237 + 82 = 319
    s = run(valve_task, [("set_stations", {"n_stations": 5}), assign(sub), ("submit_balance", {"status": "complete", "reported_n_stations": 5, "reported_cycle_time": 240})])
    assert s["feasible"] and not s["optimal"] and s["gap"] == 79
    assert s["report_consistent"] is False
    assert "report_mismatch" in s["labels"]


def test_silent_error_on_infeasible_balance(valve_task):
    bad = dict(OPT)
    bad[3] = 1  # 3 must follow 12 (station 3)
    s = run(valve_task, [("set_stations", {"n_stations": 5}), assign(bad), ("submit_balance", {"status": "complete", "reported_n_stations": 5, "reported_cycle_time": 311})])
    assert s["silent_error"] and not s["success"]
    assert s["primary_failure"] == "constraint_violation"
    assert "no_validation" in s["labels"]


def test_t4_success_and_silent_error():
    t4 = make_task(type="T4", expected="infeasible", spec={"problem": "SALBP-1", "cycle_time": 239, "max_stations": 5})
    ok = run(t4, [("submit_balance", {"status": "infeasible", "explanation": "5 stations need 240 s"})])
    assert ok["success"] and not ok["silent_error"]
    bad = run(t4, [("set_stations", {"n_stations": 5}), assign(OPT), ("submit_balance", {"status": "complete"})])
    assert bad["silent_error"] and not bad["success"]


def test_t5_requires_asking_first():
    t5 = make_task(type="T5", hidden={"max_stations": 5})
    good = run(t5, [("ask_engineer", {"message": "How many stations?"}), ("set_stations", {"n_stations": 5}), assign(OPT), ("submit_balance", {"status": "complete", "reported_n_stations": 5, "reported_cycle_time": 240})])
    assert good["success"] and good["asked_first"]
    guessed = run(t5, [("set_stations", {"n_stations": 5}), assign(OPT), ("submit_balance", {"status": "complete", "reported_n_stations": 5, "reported_cycle_time": 240})])
    assert not guessed["success"] and "assumed_parameter" in guessed["labels"]


def test_no_submission_is_premature_stop(valve_task):
    s = run(valve_task, [("get_line_summary", {})], agent_end="stopped")
    assert s["primary_failure"] == "premature_stop"
