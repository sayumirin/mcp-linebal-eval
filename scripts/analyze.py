"""Rebuild every results table and figure from the saved run logs. Calls no API.

  uv run python scripts/analyze.py                         # all results/e*/runs.jsonl[.gz]
  uv run python scripts/analyze.py --runs results/ci/runs.jsonl --out results/ci/report

Outputs (in --out, default results/):
  tables/summary.md        one row per (experiment, model, variant): rates with 95% Wilson CIs,
                           pass^1 / pass^3, optimality gap, cost
  tables/by_type.md        success, silent errors and escalation per task type
  tables/failures.md       primary failure labels
  tables/paired.md         paired comparisons between variants on the same tasks
  tables/e0.md             solver check against published optima and the AssemPlanner case
  tables/runs.csv          one row per run (flat), for your own analysis
  figures/main.png         feasible and optimal rates per architecture (E1, or whatever is present)
  figures/escalation.png   silent errors and correct handling on T4/T5 (E2, if present)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from linebal.runner import read_runs, valid_runs  # noqa: E402
from linebal.stats import mcnemar_exact, mean_pass_hat_k, median_iqr, paired_bootstrap, wilson  # noqa: E402

# Reference palette (dataviz skill, light mode): categorical slots 1-3, text and surface tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def variant_label(r: dict) -> str:
    lab = r["arch"]
    if r["escalation"] != "lenient":
        lab += f"/{r['escalation']}"
    rate = r["config"]["faults"]["rate"]
    if rate:
        lab += f"/faults {rate:g}"
    return lab


def pct(x: float) -> str:
    return "–" if x != x else f"{100 * x:.1f}"


def rate_ci(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return "–" if n == 0 else f"{100 * p:.1f} [{100 * lo:.0f}, {100 * hi:.0f}]"


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n"


# By default runs are grouped by condition (model, architecture, escalation prompt, fault rate) and
# pooled across experiment labels: E1 and E2 both contain A1/lenient/no-fault runs on T1-T3, which
# are the same condition. --by-experiment keeps the experiments apart.
POOL = True


def scope(r: dict) -> str:
    return "pooled" if POOL else r["experiment"]


def group_key(r: dict) -> tuple:
    return (scope(r), r["model_label"], variant_label(r))


def summarize(runs: list[dict]) -> tuple[str, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[group_key(r)].append(r)
    rows, records = [], []
    for key in sorted(groups):
        rs = groups[key]
        bal = [r for r in rs if r["task_type"] in ("T1", "T2", "T3")]
        n_b = len(bal)
        feas = sum(r["score"]["feasible"] for r in bal)
        opt = sum(r["score"]["optimal"] for r in bal)
        near = sum(r["score"]["near_optimal"] for r in bal)
        comp = [r for r in rs if r["score"]["status"] == "complete"]
        cons = sum(bool(r["score"]["report_consistent"]) for r in comp)
        silent = sum(r["score"]["silent_error"] for r in rs)
        succ = sum(r["score"]["success"] for r in rs)
        per_task: dict[str, list[bool]] = defaultdict(list)
        for r in rs:
            per_task[r["task_id"]].append(r["score"]["success"])
        pt = [(sum(v), len(v)) for v in per_task.values()]
        n3 = sum(1 for _, n in pt if n >= 3)  # tasks with at least three runs
        gaps = [r["score"]["gap_rel"] for r in bal if r["score"]["feasible"] and r["score"]["gap_rel"] is not None]
        med, q1, q3 = median_iqr(gaps)
        tin = sum(r["score"]["cost"]["input_tokens"] for r in rs) / len(rs)
        tout = sum(r["score"]["cost"]["output_tokens"] for r in rs) / len(rs)
        calls = sum(r["score"]["cost"]["llm_calls"] for r in rs) / len(rs)
        tools = sum(r["score"]["cost"]["tool_calls"] for r in rs) / len(rs)
        p3 = mean_pass_hat_k(pt, 3)
        rows.append(
            [
                *key,
                len(rs),
                rate_ci(succ, len(rs)),
                rate_ci(feas, n_b),
                rate_ci(opt, n_b),
                pct(near / n_b) if n_b else "–",
                rate_ci(cons, len(comp)),
                rate_ci(silent, len(rs)),
                pct(mean_pass_hat_k(pt, 1)),
                "–" if p3 != p3 else f"{pct(p3)} ({n3})",
                "–" if med != med else f"{100 * med:.1f} ({100 * q1:.1f}–{100 * q3:.1f})",
                f"{tin:,.0f} / {tout:,.0f}",
                f"{calls:.1f} / {tools:.1f}",
            ]
        )
        records.append({"key": key, "n_bal": n_b, "feasible": feas, "optimal": opt, "silent": silent, "n": len(rs)})
    header = [
        "scope", "model", "variant", "runs", "success %", "feasible % (T1–T3)", "optimal % (T1–T3)",
        "near-opt %", "report consistent %", "silent error %", "pass^1 %", "pass^3 % (tasks)",
        "gap % median (IQR)", "tokens in / out", "LLM / tool calls",
    ]
    note = (
        "\nRates are per run with 95% Wilson intervals. pass^k is the unbiased estimator of tau-bench "
        "(per task, then averaged) on the success criterion; pass^3 uses only tasks with at least three runs (count in brackets). The gap is relative to the exact optimum and "
        "only counts feasible runs. 'Report consistent' is over runs that submitted a balance.\n"
    )
    return md_table(header, rows) + note, records


def by_type(runs: list[dict]) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[(*group_key(r), r["task_type"])].append(r)
    rows = []
    for key in sorted(groups):
        rs = groups[key]
        n = len(rs)
        rows.append(
            [
                *key,
                n,
                rate_ci(sum(r["score"]["success"] for r in rs), n),
                rate_ci(sum(r["score"]["silent_error"] for r in rs), n),
                rate_ci(sum(r["score"]["asked"] > 0 for r in rs), n),
                rate_ci(sum(r["score"]["false_infeasible"] for r in rs), n),
            ]
        )
    header = ["scope", "model", "variant", "type", "runs", "success %", "silent error %", "asked engineer %", "false 'infeasible' %"]
    note = (
        "\nT4 success: the conflict is reported and no balance is submitted. T5 success: the agent asks "
        "before writing anything, then submits a feasible balance. On T1–T3 any question to the engineer is "
        "unnecessary.\n"
    )
    return md_table(header, rows) + note


def failures(runs: list[dict]) -> str:
    groups: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    labels: set[str] = set()
    for r in runs:
        f = r["score"]["primary_failure"]
        if f in ("ok",):
            continue
        groups[group_key(r)][f] += 1
        labels.add(f)
    cols = sorted(labels)
    rows = [[*k, *(groups[k].get(c, 0) for c in cols)] for k in sorted(groups)]
    return md_table(["scope", "model", "variant", *cols], rows) if rows else "No failed runs.\n"


def paired(runs: list[dict]) -> str:
    """Compare variants of the same experiment and model on the tasks both ran."""
    by: dict[tuple, dict[str, dict[str, list[bool]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in runs:
        by[(scope(r), r["model_label"])][variant_label(r)][r["task_id"]].append(r["score"]["success"])
    rows = []
    for (exp, model), variants in sorted(by.items()):
        names = sorted(variants)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                common = sorted(set(variants[a]) & set(variants[b]))
                if not common:
                    continue
                diffs = [sum(variants[b][t]) / len(variants[b][t]) - sum(variants[a][t]) / len(variants[a][t]) for t in common]
                mean, lo, hi = paired_bootstrap(diffs)
                # McNemar on (task, repeat) pairs
                bb = cc = 0
                for t in common:
                    for x, y in zip(variants[a][t], variants[b][t]):
                        bb += x and not y
                        cc += y and not x
                rows.append([exp, model, f"{b} − {a}", len(common), f"{100 * mean:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]", f"{bb}/{cc}", f"{mcnemar_exact(bb, cc):.3g}"])
    header = ["scope", "model", "difference in success (pp)", "tasks", "mean [95% bootstrap CI]", "discordant a/b", "McNemar p"]
    note = "\nDifferences are in percentage points of per-task success rate; the interval resamples tasks. McNemar pairs runs by task and repeat index.\n"
    return md_table(header, rows) + note if rows else "No paired comparisons (need at least two variants on the same tasks).\n"


def e0_table() -> str:
    parts = []
    bench = ROOT / "results" / "e0" / "benchmark.csv"
    if bench.exists():
        with bench.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        verdicts: dict[str, int] = defaultdict(int)
        for r in rows:
            verdicts[r["verdict"]] += 1
        secs = sorted(float(r["seconds"]) for r in rows if r["status"] == "optimal")
        med = secs[len(secs) // 2] if secs else float("nan")
        rpw_opt = sum(1 for r in rows if r["status"] == "optimal" and r["rpw"] and int(r["rpw"]) == int(r["objective"]))
        n_opt = sum(1 for r in rows if r["status"] == "optimal")
        parts.append(
            f"Exact CP-SAT solver vs published optima (Scholl 1993 data sets, graphs with at most 60 tasks): "
            f"{len(rows)} instances; " + ", ".join(f"{v} {k}" for k, v in sorted(verdicts.items()))
            + f". Median solve time of proven optima {med:.2f} s. The RPW heuristic reaches the optimum on {rpw_opt} of {n_opt}.\n\n"
        )
        unproven = [r for r in rows if r["verdict"] != "match"]
        if unproven:
            parts.append(md_table(["graph", "problem", "given", "published", "ours", "status"], [[r["graph"], r["problem"], r["given"], f"{r['published_lb']}–{r['published_ub']}" if r["published_lb"] != r["published_ub"] else r["published_lb"], r["objective"], r["status"]] for r in unproven]) + "\n")
    case = ROOT / "results" / "e0" / "assemplanner.json"
    if case.exists():
        c = json.loads(case.read_text(encoding="utf-8"))
        rows = [[m, v["stations"], v["cycle_time"], f"{100 * v['lbr']:.1f}"] for m, v in c["reported"].items()]
        rows.append(["**exact optimum (ours)**", c["stations"], f"**{c['optimal_cycle_time']}**", f"{100 * c['line_balancing_rate']:.1f}"])
        work = sum(c["station_loads"].values())
        rows.append(["RPW heuristic (ours)", c["stations"], c["rpw_cycle_time"], f"{100 * work / (c['stations'] * c['rpw_cycle_time']):.1f}"])
        parts.append("AssemPlanner valve case (12 operations; stations and cycle times as reported in Zhang et al. 2026, Table 3):\n\n")
        parts.append(md_table(["method", "stations", "cycle time (s)", "LBR % (recomputed)"], rows))
        parts.append("\nLBR = total work content / (stations x cycle time). The paper prints 69.3% for IGA and 79.0% for GPT-4, which do not follow from its own formula.\n")
        parts.append(f"\nOptimal station loads: {c['station_loads']}. {c['note']}\n")
    return "".join(parts) or "E0 results not found; run scripts/solve_reference.py --benchmark.\n"


def flat_rows(runs: list[dict]) -> list[dict]:
    out = []
    for r in runs:
        s = r["score"]
        out.append(
            {
                "run_id": r["run_id"], "experiment": r["experiment"], "model": r["model_label"], "variant": variant_label(r),
                "task_id": r["task_id"], "type": r["task_type"], "repeat": r["repeat"], "success": s["success"],
                "status": s["status"], "end": s["end"], "feasible": s["feasible"], "optimal": s["optimal"],
                "gap_rel": s["gap_rel"], "report_consistent": s["report_consistent"], "silent_error": s["silent_error"],
                "asked": s["asked"], "verifier_rounds": s["verifier_rounds"], "primary_failure": s["primary_failure"],
                "input_tokens": s["cost"]["input_tokens"], "output_tokens": s["cost"]["output_tokens"],
                "llm_calls": s["cost"]["llm_calls"], "tool_calls": s["cost"]["tool_calls"], "wall_time_s": r["wall_time_s"],
            }
        )
    return out


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bar_figure(groups: list[str], series: dict[str, list[tuple[int, int]]], title: str, ylabel: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    k = len(series)
    width = 0.8 / k
    for j, (name, vals) in enumerate(series.items()):
        xs = [i + (j - (k - 1) / 2) * width for i in range(len(groups))]
        stats = [wilson(c, n) if n else (float("nan"),) * 3 for c, n in vals]
        ps = [100 * s[0] for s in stats]
        # clamp float noise at p = 0 or 1 (Wilson bounds can sit 1e-17 on the wrong side)
        err = [[max(0.0, 100 * (s[0] - s[1])) if s[0] == s[0] else 0.0 for s in stats], [max(0.0, 100 * (s[2] - s[0])) if s[0] == s[0] else 0.0 for s in stats]]
        ax.bar(xs, ps, width=width - 0.03, color=SERIES[j], label=name, zorder=2)
        ax.errorbar(xs, ps, yerr=err, fmt="none", ecolor=TEXT_2, elinewidth=1, capsize=3, zorder=3)
        for x, p, s in zip(xs, ps, stats):
            if p == p:
                ax.text(x, 100 * s[2] + 2, f"{p:.0f}", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, color=TEXT)
    ax.set_ylim(0, 108)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11, pad=22)
    ax.legend(frameon=False, loc="lower left", ncol=k, fontsize=8, labelcolor=TEXT, bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


OUTCOMES = [  # (label, colour): reference palette slots 1, 3, 2, 7 - validated as an adjacent sequence
    ("optimal", "#2a78d6"),
    ("feasible, not optimal", "#1baf7a"),
    ("no answer (reply hit the output limit)", "#eb6834"),
    ("submitted an infeasible balance", "#6250d6"),
]


def outcome_of(r: dict) -> str:
    s = r["score"]
    if s["optimal"]:
        return OUTCOMES[0][0]
    if s["feasible"]:
        return OUTCOMES[1][0]
    if s["status"] == "complete":
        return OUTCOMES[3][0]
    return OUTCOMES[2][0]  # every non-submitting T1-T3 run in our logs hit the output limit; others are labelled in failures.md


def outcome_figure(runs: list[dict], path: Path) -> bool:
    """Two panels of 100% stacked bars on T1-T3: by condition and by task type."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bal = [r for r in runs if r["task_type"] in ("T1", "T2", "T3")]
    conds = [v for v in sorted({variant_label(r) for r in bal}) if sum(variant_label(r) == v for r in bal) >= MIN_N]
    if not conds:
        return False
    panels = [
        ("By setup", [(ARCH_NAMES.get(v, v), [r for r in bal if variant_label(r) == v]) for v in conds]),
        ("By work order, all setups", [({"T1": "Cycle time given", "T2": "Station count given", "T3": "Plus zoning constraints"}[t], [r for r in bal if r["task_type"] == t]) for t in ("T1", "T2", "T3")]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4), dpi=160, gridspec_kw={"wspace": 0.8})
    fig.patch.set_facecolor(SURFACE)
    for ax, (title, rows) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(TEXT_2)
        ax.tick_params(colors=TEXT_2, length=0)
        for y, (label, rs) in enumerate(reversed(rows)):
            left = 0.0
            for name, colour in OUTCOMES:
                k = sum(outcome_of(r) == name for r in rs)
                w = 100 * k / len(rs)
                if w > 0:
                    ax.barh(y, w - 0.4, left=left + 0.2, height=0.62, color=colour)  # small gaps between segments
                    if w >= 6:
                        ax.text(left + w / 2, y, str(k), ha="center", va="center", fontsize=8, color="white")
                left += w
            ax.text(101.5, y, f"n={len(rs)}", va="center", fontsize=8, color=TEXT_2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([lab for lab, _ in reversed(rows)], color=TEXT, fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of runs", color=TEXT_2, fontsize=9)
        ax.set_title(title, loc="left", color=TEXT, fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in OUTCOMES]
    n_bad = sum(outcome_of(r) == OUTCOMES[3][0] for r in bal)
    labels = [n if n != OUTCOMES[3][0] else f"{n} ({n_bad})" for n, _ in OUTCOMES]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=TEXT, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Outcome of {len(bal)} balancing runs", x=0.01, ha="left", color=TEXT, fontsize=11)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.82, bottom=0.27)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


ARCH_NAMES = {"A0": "Direct prompting", "A1": "Agent with MCP tools", "A2": "Agent with verifier", "A1/faults 0.1": "Agent, 10% tool errors"}  # figure labels
MIN_N = 10  # conditions with fewer runs stay in the tables but are not drawn


def figures(runs: list[dict], out: Path) -> list[Path]:
    made = []
    variants: list[str] = []
    exps = sorted({r["experiment"] for r in runs})
    main_exp = "all experiments" if POOL else ("e1" if "e1" in exps else exps[0])
    rs = [r for r in runs if (POOL or r["experiment"] == main_exp) and r["task_type"] in ("T1", "T2", "T3")]
    for model in sorted({r["model_label"] for r in rs}):
        sub = [r for r in rs if r["model_label"] == model]
        variants = [v for v in sorted({variant_label(r) for r in sub}) if sum(variant_label(r) == v for r in sub) >= MIN_N]
        if not variants:
            continue
        feas, opt = [], []
        for v in variants:
            vv = [r for r in sub if variant_label(r) == v]
            feas.append((sum(r["score"]["feasible"] for r in vv), len(vv)))
            opt.append((sum(r["score"]["optimal"] for r in vv), len(vv)))
        labels = [ARCH_NAMES.get(v, v) for v in variants]
        name = "main.png" if model == sorted({r["model_label"] for r in rs})[0] else f"main_{model}.png"
        bar_figure(labels, {"feasible": feas, "optimal": opt}, f"Feasible and optimal balances, T1–T3 ({model}, {main_exp})", "% of runs (95% CI)", out / "figures" / name)
        made.append(out / "figures" / name)
    if POOL and outcome_figure(runs, out / "figures" / "outcomes.png"):
        made.append(out / "figures" / "outcomes.png")
    esc = [r for r in runs if r["task_type"] in ("T4", "T5")]
    if esc:
        exp = "all experiments" if POOL else ("e2" if any(r["experiment"] == "e2" for r in esc) else esc[0]["experiment"])
        esc = [r for r in esc if POOL or r["experiment"] == exp]
        model = sorted({r["model_label"] for r in esc})[0]
        esc = [r for r in esc if r["model_label"] == model]
        variants = [v for v in sorted({variant_label(r) for r in esc}) if sum(variant_label(r) == v for r in esc) >= MIN_N]
    if esc and variants:
        series: dict[str, list[tuple[int, int]]] = {"silent error, T4": [], "silent error, T5": [], "correct handling, T4+T5": []}
        for v in variants:
            vv = [r for r in esc if variant_label(r) == v]
            t4 = [r for r in vv if r["task_type"] == "T4"]
            t5 = [r for r in vv if r["task_type"] == "T5"]
            series["silent error, T4"].append((sum(r["score"]["silent_error"] for r in t4), len(t4)))
            series["silent error, T5"].append((sum(r["score"]["silent_error"] for r in t5), len(t5)))
            series["correct handling, T4+T5"].append((sum(r["score"]["success"] for r in vv), len(vv)))
        bar_figure(variants, series, f"Infeasible (T4) and underspecified (T5) work orders ({model}, {exp})", "% of runs (95% CI)", out / "figures" / "escalation.png")
        made.append(out / "figures" / "escalation.png")
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--by-experiment", action="store_true", help="keep experiments apart instead of pooling by condition")
    a = ap.parse_args()
    global POOL
    POOL = not a.by_experiment
    # pilot runs (same configuration) first, so a full-run record wins when both exist for a run id;
    # crashed_api_errors.jsonl holds API-refused episodes and is reported, not scored
    paths = a.runs or [p for e in sorted((ROOT / "results").glob("e*/")) for name in ("pilot.jsonl*", "runs.jsonl*") for p in sorted(e.glob(name))]
    tables = a.out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "e0.md").write_text(e0_table(), encoding="utf-8", newline="\n")
    if not paths:
        print("no run logs found; wrote tables/e0.md only")
        return
    runs, dropped = valid_runs(read_runs(paths))
    summary, _ = summarize(runs)
    if dropped:
        summary += f"\n{dropped} logged records were left out: crashed episodes (API or harness errors) and earlier attempts superseded by a rerun.\n"
    (tables / "summary.md").write_text(summary, encoding="utf-8", newline="\n")
    (tables / "by_type.md").write_text(by_type(runs), encoding="utf-8", newline="\n")
    (tables / "failures.md").write_text(failures(runs), encoding="utf-8", newline="\n")
    (tables / "paired.md").write_text(paired(runs), encoding="utf-8", newline="\n")
    rows = flat_rows(runs)
    with (tables / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")  # LF on every platform
        w.writeheader()
        w.writerows(rows)
    made = figures(runs, a.out)
    print(f"{len(runs)} runs from {len(paths)} log(s) -> {tables}; figures: {[str(p.name) for p in made]}")


if __name__ == "__main__":
    main()
