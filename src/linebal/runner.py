"""Run episodes and write one JSON line per run.

Every record carries what is needed to recompute the result tables without calling any API: the
model and its settings, prompt version, seeds, every LLM response with token usage, every tool call
with its result, the submissions and the score.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from . import __version__
from .agents import ARCHITECTURES
from .agents.base import McpSession, RunConfig
from .agents.llm import LLM, BalanceError
from .agents.prompts import PROMPT_VERSION
from .faults import FaultConfig
from .scoring import score_run
from .tasks import Task, build_instance


def run_id(exp: str, model: str, arch: str, escalation: str, fault_rate: float, task_id: str, rep: int) -> str:
    return f"{exp}|{model}|{arch}|{escalation}|f{fault_rate:g}|{task_id}|r{rep}"


def run_seed(rid: str) -> int:
    return int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16)


async def run_episode(task: Task, llm: LLM, cfg: RunConfig, ref: dict | None, meta: dict) -> dict:
    t0 = time.time()
    agent: dict = {"llm_calls": [], "events": [], "end": "crash"}
    export: dict = {"submissions": [], "calls": [], "balance": {"n_stations": 0, "assignment": {}}}
    error = None
    balance_error: BalanceError | None = None
    try:
        async with McpSession(task, cfg) as session:
            try:
                agent = await ARCHITECTURES[cfg.arch](session, llm, cfg)
            except BalanceError as e:
                # keep it out of the MCP client's task group, which would wrap it in an ExceptionGroup
                balance_error = e
            except Exception:  # noqa: BLE001 - a crashed episode is a data point, not a reason to stop
                error = traceback.format_exc(limit=8)
            if balance_error is None:
                export = await session.export()
    except Exception:  # noqa: BLE001
        error = (error or "") + traceback.format_exc(limit=8)
    if balance_error is not None:
        raise balance_error
    agent["arch"] = cfg.arch
    if error:
        agent["end"] = "crash"
    inst = build_instance(task)
    score = score_run(task, inst, ref, export, agent)
    return {
        **meta,
        "task_id": task.task_id,
        "task_type": task.type,
        "arch": cfg.arch,
        "escalation": cfg.escalation,
        "config": cfg.to_dict(),
        "prompt_version": PROMPT_VERSION,
        "linebal_version": __version__,
        "model": llm.describe(),
        # model aliases such as "deepseek-flash" can move to a new version, so record when it ran
        "started_at": datetime.fromtimestamp(t0, timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": round(time.time() - t0, 2),
        "score": score,
        "agent": agent,
        "env": export,
        "error": error,
    }


def open_text(path: Path, mode: str = "rt"):
    return gzip.open(path, mode, encoding="utf-8") if path.suffix == ".gz" else path.open(mode.replace("t", ""), encoding="utf-8")


def read_runs(paths: Iterable[Path]) -> list[dict]:
    out = []
    for p in paths:
        with open_text(Path(p)) as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


def valid_runs(runs: list[dict]) -> tuple[list[dict], int]:
    """Latest record per run id, without crashed episodes. Returns (runs, number dropped)."""
    latest: dict[str, dict] = {}
    for r in runs:
        latest[r["run_id"]] = r
    keep = [r for r in latest.values() if not r.get("error")]
    return keep, len(runs) - len(keep)


async def run_experiment(
    exp: str,
    tasks: list[Task],
    references: dict[str, dict],
    models: list[tuple[str, LLM]],
    variants: list[RunConfig],
    repeats: int,
    out_path: Path,
    concurrency: int = 4,
    progress: bool = True,
) -> int:
    """Run every (model, variant, task, repeat) not yet in `out_path`; append records as they finish."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        done = {r["run_id"] for r in valid_runs(read_runs([out_path]))[0]}  # crashed runs are retried
    # Queue repeat-major, then task, then condition: if a run stops early (budget, rate limit), every
    # condition has covered the same tasks so far. (A condition-major queue left the 2026-11-08 runs
    # with A0 on 26 tasks and A2 on none.)
    jobs = []
    for rep in range(repeats):
        for task in tasks:
            for model_label, llm in models:
                for cfg in variants:
                    rid = run_id(exp, model_label, cfg.arch, cfg.escalation, cfg.faults.rate, task.task_id, rep)
                    if rid not in done:
                        jobs.append((rid, model_label, llm, cfg, task, rep))
    if progress:
        print(f"{exp}: {len(done)} valid runs already logged, {len(jobs)} to go", flush=True)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    finished = 0
    stopped: list[str] = []

    async def one(rid: str, model_label: str, llm: LLM, cfg: RunConfig, task: Task, rep: int) -> None:
        nonlocal finished
        seed = run_seed(rid)
        run_llm = llm.for_run(seed) if hasattr(llm, "for_run") else llm
        run_cfg = replace(cfg, faults=FaultConfig(cfg.faults.rate, list(cfg.faults.kinds), seed))
        meta = {"run_id": rid, "experiment": exp, "model_label": model_label, "repeat": rep, "seed": seed}
        async with sem:
            if stopped:
                return
            try:
                rec = await run_episode(task, run_llm, run_cfg, references.get(task.task_id), meta)
            except BalanceError as e:
                if not stopped and progress:
                    print(f"STOPPED: the provider reports an exhausted balance ({str(e)[:160]})", flush=True)
                stopped.append(rid)
                return
        async with lock:
            with open_text(out_path, "at") as f:
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            finished += 1
            if progress:
                s = rec["score"]
                print(f"[{finished}/{len(jobs)}] {rid} success={s['success']} end={s['end']} tokens={s['cost']['input_tokens']}+{s['cost']['output_tokens']}", flush=True)

    await asyncio.gather(*(one(*j) for j in jobs))
    if stopped and progress:
        print(f"{exp}: stopped early on a balance error; {len(jobs) - finished} runs not done. Top up and rerun to resume.", flush=True)
    return finished
