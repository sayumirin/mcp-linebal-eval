"""Shared agent plumbing: an MCP session for one episode and the run configuration.

Every architecture talks to the environment only through the MCP server, and all of them see the
same public tool list, so the comparison is about the architecture and not about tool access.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from ..env import LineEnv
from ..faults import FaultConfig
from ..server import build_server
from ..tasks import Task


@dataclass
class RunConfig:
    arch: str  # A0 | A1 | A2
    escalation: str = "lenient"  # lenient | strict
    temperature: float = 0.0
    max_tokens: int = 4096
    max_steps: int = 50  # LLM calls per episode (A1, A2)
    max_rounds: int = 10  # A2 verifier rounds before handing over to the engineer
    transport: str = "inproc"  # inproc | stdio
    faults: FaultConfig = field(default_factory=FaultConfig)
    tasks_path: str = ""  # needed for the stdio transport

    def to_dict(self) -> dict:
        d = asdict(self)
        d["faults"] = self.faults.to_dict()
        return d


def _inline_refs(schema: dict) -> dict:
    """Resolve local $ref/$defs and drop pydantic titles: some OpenAI-compatible endpoints reject $ref."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                return walk(copy.deepcopy(defs[name]))
            return {k: walk(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


class McpSession:
    """Opens the MCP server for one task (in-process or as a stdio subprocess)."""

    def __init__(self, task: Task, cfg: RunConfig):
        self.task = task
        self.cfg = cfg
        self.client: Client | None = None
        self.tools: list = []

    async def __aenter__(self) -> McpSession:
        if self.cfg.transport == "stdio":
            f = self.cfg.faults
            tasks_file = Path(self.cfg.tasks_path)
            if not tasks_file.is_absolute() and not tasks_file.exists():
                tasks_file = Path(__file__).resolve().parents[3] / tasks_file  # relative to the repo root

            params = StdioServerParameters(
                command=sys.executable,
                args=[
                    "-m", "linebal.server",
                    "--tasks", str(tasks_file.resolve()),
                    "--task-id", self.task.task_id,
                    "--fault-rate", str(f.rate),
                    "--fault-kinds", ",".join(f.kinds),
                    "--fault-seed", str(f.seed),
                ],
            )
            self.client = Client(params)
        else:
            self.client = Client(build_server(LineEnv(self.task, faults=self.cfg.faults)))
        await self.client.__aenter__()
        listing = await self.client.list_tools()
        self.tools = [t for t in listing.tools if not t.name.startswith("admin_")]
        return self

    async def __aexit__(self, *exc) -> None:
        if self.client is not None:
            await self.client.__aexit__(*exc)

    def openai_tools(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": t.name, "description": t.description or "", "parameters": _inline_refs(t.input_schema)}}
            for t in self.tools
        ]

    @property
    def tool_names(self) -> set[str]:
        return {t.name for t in self.tools}

    async def call(self, name: str, args: dict | None) -> tuple[str, bool]:
        """Agent-visible tool call. Returns (text, is_error)."""
        assert self.client is not None
        if name not in self.tool_names:
            return json.dumps({"error": {"code": "unknown_tool", "message": f"No tool named {name!r}"}}), True
        if args is None:
            return json.dumps({"error": {"code": "bad_arguments", "message": "Arguments were not valid JSON"}}), True
        r = await self.client.call_tool(name, args)
        text = "\n".join(getattr(c, "text", "") for c in r.content)
        return text, bool(r.is_error)

    async def admin(self, tool: str, args: dict | None = None) -> dict:
        """Harness-side call: no fault injection, not counted as an agent action."""
        assert self.client is not None
        r = await self.client.call_tool("admin_call", {"tool": tool, "arguments": args or {}})
        return json.loads(r.content[0].text)

    async def export(self) -> dict:
        assert self.client is not None
        r = await self.client.call_tool("admin_export", {})
        return json.loads(r.content[0].text)


def parse_json_answer(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply (handles ```json fences and chatter)."""
    if not text:
        return None
    s = text.strip()
    if "```" in s:
        parts = s.split("```")
        for p in parts[1::2]:
            p = p.strip()
            if p.lower().startswith("json"):
                p = p[4:]
            try:
                v = json.loads(p)
                if isinstance(v, dict):
                    return v
            except json.JSONDecodeError:
                continue
    start = s.find("{")
    while start != -1:
        depth = 0
        for k in range(start, len(s)):
            if s[k] == "{":
                depth += 1
            elif s[k] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(s[start : k + 1])
                        if isinstance(v, dict):
                            return v
                    except json.JSONDecodeError:
                        break
                    break
        start = s.find("{", start + 1)
    return None
