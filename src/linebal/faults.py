"""Tool fault injection (experiment E3).

Each environment tool call may fail with probability `rate`. Faults are drawn from a seeded RNG so a
run is reproducible. Kinds:

  transient     the call is not executed and returns a "temporarily unavailable" error (MVP)
  timeout       the call is not executed and returns a timeout error
  partial       read tools return a truncated list, flagged as `partial: true`
  write_reject  write tools are refused with a lock error; nothing is changed
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

READ_TOOLS = {"get_line_summary", "get_operations", "get_constraints", "get_balance", "validate_balance", "compute_kpis"}
WRITE_TOOLS = {"set_stations", "assign_operations"}
NEVER_FAULT = {"submit_balance", "ask_engineer"}


@dataclass
class FaultConfig:
    rate: float = 0.0
    kinds: list[str] = field(default_factory=lambda: ["transient"])
    seed: int = 0

    def to_dict(self) -> dict:
        return {"rate": self.rate, "kinds": list(self.kinds), "seed": self.seed}


class FaultInjector:
    def __init__(self, cfg: FaultConfig | None):
        self.cfg = cfg or FaultConfig()
        self.rng = random.Random(self.cfg.seed)
        self.injected: list[dict] = []

    def draw(self, tool: str) -> str | None:
        """Return the fault kind to inject for this call, or None."""
        if self.cfg.rate <= 0 or tool in NEVER_FAULT:
            return None
        if self.rng.random() >= self.cfg.rate:
            return None
        kinds = [k for k in self.cfg.kinds if self._applies(k, tool)]
        if not kinds:
            return None
        kind = self.rng.choice(kinds)
        self.injected.append({"tool": tool, "kind": kind})
        return kind

    @staticmethod
    def _applies(kind: str, tool: str) -> bool:
        if kind == "partial":
            return tool in {"get_operations", "get_balance"}
        if kind == "write_reject":
            return tool in WRITE_TOOLS
        return True
