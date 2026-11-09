"""Model interface. Messages use the OpenAI chat format throughout.

* `OpenAICompatLLM` talks to any OpenAI-compatible endpoint: DeepSeek (https://api.deepseek.com),
  OpenAI, OpenRouter, or a vLLM server on Modal.
* `FakeLLM` (in fake.py) is a deterministic scripted policy used by tests and CI, so the whole
  pipeline runs without any API key.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


class BalanceError(RuntimeError):
    """The provider refused the request because the account balance is too low."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None  # None if the model produced invalid JSON
    raw_arguments: str = ""


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)  # input_tokens, output_tokens, plus provider extras
    model: str = ""
    finish_reason: str = ""
    latency_s: float = 0.0
    reasoning: str | None = None

    def to_log(self) -> dict:
        d = asdict(self)
        d["latency_s"] = round(self.latency_s, 3)
        return d

    def assistant_message(self, include_reasoning: bool = False) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if include_reasoning and self.reasoning:
            # DeepSeek thinking mode expects its reasoning back within one tool-calling turn
            msg["reasoning_content"] = self.reasoning
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.raw_arguments or json.dumps(tc.arguments or {})}}
                for tc in self.tool_calls
            ]
        return msg


class LLM(ABC):
    name: str = "llm"
    send_reasoning: bool = False  # echo reasoning_content back in later turns

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...

    def describe(self) -> dict:
        return {"name": self.name}


class OpenAICompatLLM(LLM):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        extra_body: dict | None = None,
        timeout: float = 300.0,
        max_retries: int = 6,
        note: str = "",
        send_reasoning: bool = False,
    ):
        from openai import AsyncOpenAI

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"environment variable {api_key_env} is not set")
        self.name = model
        self.base_url = base_url
        self.extra_body = extra_body or {}
        self.note = note
        self.send_reasoning = send_reasoning
        self.client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=max_retries)

    def describe(self) -> dict:
        return {"name": self.name, "base_url": self.base_url, "extra_body": self.extra_body, "note": self.note, "send_reasoning": self.send_reasoning}

    async def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self.name, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            kwargs["tools"] = tools
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        t0 = time.perf_counter()
        try:
            r = await self.client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            # An exhausted account is an infrastructure failure, not an agent failure: stop the
            # experiment instead of logging crashed episodes.
            status = getattr(e, "status_code", None)
            text = str(e).lower()
            if status == 402 or "top up" in text or "insufficient balance" in text:
                raise BalanceError(str(e)) from e
            raise
        latency = time.perf_counter() - t0
        choice = r.choices[0]
        msg = choice.message
        calls = []
        for tc in msg.tool_calls or []:
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw) if raw.strip() else {}
                if not isinstance(args, dict):
                    args = None
            except json.JSONDecodeError:
                args = None
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args, raw_arguments=raw))
        usage: dict[str, Any] = {}
        if r.usage is not None:
            u = r.usage.model_dump()
            usage = {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}
            for k in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
                if k in u:
                    usage[k] = u[k]
            details = u.get("completion_tokens_details") or {}
            if details.get("reasoning_tokens"):
                usage["reasoning_tokens"] = details["reasoning_tokens"]
        return LLMResponse(
            content=msg.content or "",
            tool_calls=calls,
            usage=usage,
            model=r.model,
            finish_reason=choice.finish_reason or "",
            latency_s=latency,
            reasoning=getattr(msg, "reasoning_content", None),
        )


def make_llm(cfg: dict) -> LLM:
    """Build a model from a config entry, e.g.
    {provider: openai_compat, model: deepseek-chat, base_url: https://api.deepseek.com, api_key_env: DEEPSEEK_API_KEY}
    {provider: fake, seed: 0, profile: sloppy}
    """
    provider = cfg.get("provider", "openai_compat")
    if provider == "fake":
        from .fake import FakeLLM

        return FakeLLM(seed=cfg.get("seed", 0), profile=cfg.get("profile", "sloppy"))
    if provider == "openai_compat":
        return OpenAICompatLLM(
            model=cfg["model"],
            base_url=cfg.get("base_url"),
            api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
            extra_body=cfg.get("extra_body"),
            note=cfg.get("note", ""),
            send_reasoning=bool(cfg.get("send_reasoning", False)),
        )
    raise ValueError(f"unknown provider {provider!r}")
