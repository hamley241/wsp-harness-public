"""Pluggable LLM client: Anthropic backend (lazy import), OpenRouter (urllib) and an offline MockLLM."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from wsp.sandbox import mock_constructor_source

_FENCE = re.compile(r"```([^\s`]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_ROLE = re.compile(r"^ROLE:\s*([\w-]+)\s*$", re.MULTILINE)
_HIST_LINE = re.compile(r"^- length (\d+): (\d+)$", re.MULTILINE)
_WEIGHT_LINE = re.compile(r"^(Wx|Wh|bh|w|c)\[(\d+)\] = (\S+)$", re.MULTILINE)
_NUMERIC_EDITS = 8


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class LLMError(RuntimeError):
    """The model returned no usable text (empty choices, null content, non-text stop)."""

    def __init__(self, message: str, *, finish_reason=None, native_finish_reason=None, refusal_text=None, id=None):
        super().__init__(message)
        self.finish_reason = finish_reason
        self.native_finish_reason = native_finish_reason
        self.refusal_text = refusal_text
        self.id = id


class LLMRefusal(LLMError):
    """The provider refused the request (content filter) rather than failing it."""


class BudgetStop(RuntimeError):
    """Accrued spend plus the next call's reserve would exceed the cap. Never retried: it fails the seed."""


class AnthropicClient:
    def __init__(self, model: str = "claude-fable-5-1", max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens
        self.last_usage: dict | None = None
        self.usage_log: list[dict] = []

    def complete(self, system: str, user: str) -> str:
        import anthropic  # optional dependency group `llm`; key from ANTHROPIC_API_KEY

        client = anthropic.Anthropic()
        msg = client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                     messages=[{"role": "user", "content": user}])
        usage = getattr(msg, "usage", None)
        stop = getattr(msg, "stop_reason", None)
        # The Messages API does not break out reasoning tokens; record 0 so totals stay numeric.
        self.last_usage = {"prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
                           "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
                           "reasoning_tokens": 0, "id": getattr(msg, "id", None), "finish_reason": stop}
        self.usage_log.append(self.last_usage)
        text = "".join(b.text for b in (msg.content or []) if getattr(b, "type", None) == "text")
        if stop == "refusal":
            raise LLMRefusal(f"Anthropic refusal (stop_reason={stop})", finish_reason=stop,
                             native_finish_reason=stop, id=self.last_usage["id"])
        if not text:
            raise LLMError(f"Anthropic returned no text block (stop_reason={stop})", finish_reason=stop,
                           native_finish_reason=stop, id=self.last_usage["id"])
        return text


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NO_KEY_MSG = "no OpenRouter key: set OPENROUTER_API_KEY"
_RETRY_BACKOFF_S = (2, 4, 8)


def _urllib_transport(url: str, headers: dict, body: bytes, timeout_s: float):
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class OpenRouterClient:
    """OpenRouter chat completions over urllib. The key is resolved at construction and never surfaced."""

    def __init__(self, model: str = "anthropic/claude-fable-5.1", max_tokens: int = 20000,
                 reasoning_max_tokens: int = 16000, timeout_s: float = 600, api_key: str | None = None,
                 transport=None):
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_max_tokens = reasoning_max_tokens
        self.timeout_s = timeout_s
        self._transport = transport or (lambda url, headers, body: _urllib_transport(url, headers, body, timeout_s))
        self._key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._key:
            raise RuntimeError(NO_KEY_MSG)
        self.last_usage: dict | None = None
        self.usage_log: list[dict] = []

    def complete(self, system: str, user: str) -> str:
        payload = {"model": self.model, "max_tokens": self.max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.reasoning_max_tokens:
            payload["reasoning"] = {"max_tokens": self.reasoning_max_tokens}
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                   "HTTP-Referer": "https://github.com/REPLACE-WITH-PUBLIC-REPO", "X-Title": "wsp-harness"}
        body = json.dumps(payload).encode()
        for attempt in range(len(_RETRY_BACKOFF_S) + 1):
            try:
                status, raw = self._transport(OPENROUTER_URL, headers, body)
            except Exception as e:  # noqa: BLE001 - transport errors may quote the request, key included
                # `from None` drops the chain so the unsanitized original never reaches a traceback.
                raise RuntimeError(f"OpenRouter transport error: {self._redact(f'{type(e).__name__}: {e}')}") from None
            if status == 200:
                break
            if (status == 429 or status >= 500) and attempt < len(_RETRY_BACKOFF_S):
                time.sleep(_RETRY_BACKOFF_S[attempt])
                continue
            # Redact the whole body first, then truncate: a key straddling the cut would otherwise leak a prefix.
            text = self._redact(raw.decode(errors="replace"))[:500]
            raise RuntimeError(f"OpenRouter HTTP {status}: {text}")
        data = json.loads(raw)
        usage = data.get("usage") or {}
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish, native = choice.get("finish_reason"), choice.get("native_finish_reason")
        self.last_usage = {"prompt_tokens": usage.get("prompt_tokens", 0),
                           "completion_tokens": usage.get("completion_tokens", 0),
                           "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                           "id": data.get("id"), "finish_reason": finish}
        self.usage_log.append(self.last_usage)
        content = message.get("content")
        if not content:
            # A refusal arrives as HTTP 200 with null content: content_filter / native refusal.
            kw = dict(finish_reason=finish, native_finish_reason=native, refusal_text=message.get("refusal"),
                      id=data.get("id"))
            if finish == "content_filter" or native == "refusal":
                raise LLMRefusal(f"OpenRouter refusal (finish_reason={finish}, native_finish_reason={native}): "
                                 f"{message.get('refusal')}", **kw)
            raise LLMError(f"OpenRouter returned no content (finish_reason={finish})", **kw)
        return content

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "[redacted]")



class MockLLM:
    """Deterministic offline stand-in. Editor -> seeded mock constructor; generator -> random strings."""

    def __init__(self, seed: int, h: int, n_candidates: int):
        self.h = h
        self.n_candidates = n_candidates
        self._next_seed = seed
        self._rng = random.Random(seed)

    def complete(self, system: str, user: str) -> str:
        role = role_of(system)
        if role == "editor":
            src = mock_constructor_source(self.h, self._next_seed)
            self._next_seed += 1
            return f"```python\n{src}```\n"
        if role == "editor-numeric":
            # 8 distinct in-range coordinates nudged by a small delta from the value listed in the prompt.
            current = {(p, int(i)): float(v) for p, i, v in _WEIGHT_LINE.findall(user)}
            if not current:
                sizes = {"Wx": 2 * self.h, "Wh": self.h * self.h, "bh": self.h, "w": self.h, "c": 1}
                current = {(p, i): 0.0 for p, n in sizes.items() for i in range(n)}
            picks = self._rng.sample(sorted(current), min(_NUMERIC_EDITS, len(current)))
            edits = [{"param": p, "index": i, "value": round(current[(p, i)] + self._rng.uniform(-0.1, 0.1), 6)}
                     for p, i in picks]
            return "```json\n" + json.dumps({"edits": edits}) + "\n```\n"
        if role == "generator":
            hist = {int(L): int(c) for L, c in _HIST_LINE.findall(user) if int(c) > 0} or {L: 1 for L in range(1, 13)}
            lengths = self._rng.choices(list(hist), weights=list(hist.values()), k=self.n_candidates)
            lines = ["".join(self._rng.choice("01") for _ in range(L)) for L in lengths]
            return "```strings\n" + "\n".join(lines) + "\n```\n"
        raise ValueError(f"MockLLM: no ROLE marker in system prompt ({role!r})")


def role_of(system: str) -> str:
    m = _ROLE.search(system)
    return m.group(1) if m else "unknown"


def extract_fenced(text: str, lang: str) -> str:
    blocks = _FENCE.findall(text)
    for tag, body in blocks:
        if tag == lang:
            return body
    return blocks[0][1] if blocks else text


@dataclass
class CallLog:
    calls: list[dict] = field(default_factory=list)

    def complete(self, llm: LLMClient, system: str, user: str) -> str:
        """Call the client, time it, append the entry, return the response."""
        t0 = time.monotonic()
        entry = {"role": role_of(system), "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
                 "user": user, "response": None, "finish_reason": None, "error": None}
        self.calls.append(entry)
        try:
            response = llm.complete(system, user)
        except LLMError as e:
            entry.update(finish_reason=e.finish_reason, error=str(e)[:300], elapsed_s=time.monotonic() - t0)
            raise
        entry.update(response=response, finish_reason=(getattr(llm, "last_usage", None) or {}).get("finish_reason"),
                     elapsed_s=time.monotonic() - t0)
        return response

    def dump(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.calls, f, indent=1)
