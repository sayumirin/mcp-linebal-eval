"""End-to-end: fake model -> MCP server -> scoring -> analysis, without any API key."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from linebal.agents.base import RunConfig, parse_json_answer
from linebal.agents.fake import FakeLLM
from linebal.faults import FaultConfig
from linebal.runner import read_runs, run_experiment
from linebal.tasks import load_tasks

ROOT = Path(__file__).resolve().parents[1]


def test_parse_json_answer():
    assert parse_json_answer('Here:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_answer('text {"a": {"b": 2}} more') == {"a": {"b": 2}}
    assert parse_json_answer("no json") is None


def test_all_architectures_run_and_resume(tmp_path):
    tasks = load_tasks(ROOT / "tasks" / "ci.jsonl")
    refs = json.loads((ROOT / "results" / "reference" / "ci.json").read_text(encoding="utf-8"))
    variants = [RunConfig(arch="A0"), RunConfig(arch="A1", max_steps=30), RunConfig(arch="A2", max_steps=30, faults=FaultConfig(rate=0.1))]
    out = tmp_path / "runs.jsonl"
    n = asyncio.run(run_experiment("t", tasks, refs, [("fake", FakeLLM(profile="sloppy"))], variants, 1, out, progress=False))
    assert n == len(tasks) * 3
    runs = read_runs([out])
    assert len(runs) == n and not any(r["error"] for r in runs)
    a2 = [r for r in runs if r["arch"] == "A2"]
    assert all(r["config"]["faults"]["rate"] == 0.1 for r in a2)
    # a second call finds everything logged and runs nothing
    assert asyncio.run(run_experiment("t", tasks, refs, [("fake", FakeLLM())], variants, 1, out, progress=False)) == 0


def test_perfect_fake_solves_balance_tasks(tmp_path):
    tasks = [t for t in load_tasks(ROOT / "tasks" / "ci.jsonl") if t.type in ("T1", "T2", "T3", "T4")]
    refs = json.loads((ROOT / "results" / "reference" / "ci.json").read_text(encoding="utf-8"))
    out = tmp_path / "runs.jsonl"
    asyncio.run(run_experiment("t", tasks, refs, [("fake", FakeLLM(profile="perfect"))], [RunConfig(arch="A2")], 1, out, progress=False))
    runs = read_runs([out])
    assert all(r["score"]["success"] for r in runs), [(r["task_id"], r["score"]["labels"]) for r in runs if not r["score"]["success"]]


def test_analyze_script(tmp_path):
    out = tmp_path / "runs.jsonl"
    tasks = load_tasks(ROOT / "tasks" / "ci.jsonl")
    refs = json.loads((ROOT / "results" / "reference" / "ci.json").read_text(encoding="utf-8"))
    asyncio.run(run_experiment("e1", tasks, refs, [("fake", FakeLLM())], [RunConfig(arch="A1"), RunConfig(arch="A2")], 3, out, progress=False))
    report = tmp_path / "report"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze.py"), "--runs", str(out), "--out", str(report)], check=True)
    assert (report / "tables" / "summary.md").read_text(encoding="utf-8").count("| pooled |") == 2  # one row per condition
    assert (report / "figures" / "main.png").stat().st_size > 10_000
    assert (report / "figures" / "outcomes.png").stat().st_size > 10_000
    by_exp = tmp_path / "by_exp"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze.py"), "--runs", str(out), "--out", str(by_exp), "--by-experiment"], check=True)
    assert (by_exp / "tables" / "summary.md").read_text(encoding="utf-8").count("| e1 |") == 2
