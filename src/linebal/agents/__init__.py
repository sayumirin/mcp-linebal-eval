"""Agent architectures: A0 direct prompting, A1 ReAct, A2 ReAct with an external verifier.

A3 (planner + checker, extension) is not part of the MVP.
"""

from .direct import run_direct
from .modulo import run_modulo
from .react import run_react

ARCHITECTURES = {"A0": run_direct, "A1": run_react, "A2": run_modulo}

__all__ = ["ARCHITECTURES", "run_direct", "run_react", "run_modulo"]
