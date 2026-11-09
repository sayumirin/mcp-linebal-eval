from __future__ import annotations

import pytest

from linebal.data import GRAPH_DIR, OPTIMA_CSV
from linebal.tasks import Task

needs_data = pytest.mark.skipif(
    not (GRAPH_DIR.exists() and OPTIMA_CSV.exists()),
    reason="benchmark data not downloaded (run scripts/download_data.py)",
)


def make_task(**kw) -> Task:
    base = dict(
        task_id="t",
        type="T2",
        source={"kind": "case", "name": "assemplanner_valve"},
        spec={"problem": "SALBP-2", "cycle_time": None, "max_stations": 5},
        request="Use 5 stations and minimise the cycle time.",
        line_name="Valve line",
    )
    base.update(kw)
    return Task(**base)


@pytest.fixture
def valve_task() -> Task:
    return make_task()
