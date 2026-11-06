import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from wsp import llm as llm_mod
from wsp.llm import (NO_KEY_MSG, AnthropicClient, CallLog, LLMError, LLMRefusal, MockLLM, OpenRouterClient,
                     extract_fenced, role_of)
from wsp.prompts import editor_system, generator_system, generator_user
from wsp.sandbox import audit

REPO = Path(__file__).resolve().parents[1]


def test_mock_is_deterministic_and_role_aware():
    gen_user = generator_user([("01", 1)], [], {2: 3, 5: 1})
    outs = []
    for _ in range(2):
        m = MockLLM(seed=3, h=4, n_candidates=6)
        outs.append([m.complete(editor_system(), "x"), m.complete(generator_system(), gen_user),
                     m.complete(editor_system(), "x")])
    assert outs[0] == outs[1]
    e1, g, e2 = outs[0]
    assert e1 != e2  # editor seed advances per call
    assert audit(extract_fenced(e1, "python"), []).ok
    lines = extract_fenced(g, "strings").split()
    assert len(lines) == 6 and all(set(s) <= {"0", "1"} and len(s) in (2, 5) for s in lines)


def test_extract_fenced_and_role():
    assert extract_fenced("a\n```strings\n01\n10\n```\n```python\nx=1\n```", "python") == "x=1\n"
    assert extract_fenced("```strings\n01\n```", "python") == "01\n"  # fallback: first block of any tag
    assert extract_fenced("no fence", "python") == "no fence"
    assert role_of(editor_system()) == "editor" and role_of(generator_system()) == "generator"
    assert role_of("hi") == "unknown"


def test_call_log_records_and_dumps(tmp_path):
    log = CallLog()
    assert log.complete(MockLLM(1, 4, 2), editor_system(), "u").startswith("```python")
    (entry,) = log.calls
    assert entry["role"] == "editor" and entry["user"] == "u" and len(entry["system_sha256"]) == 64
    assert entry["elapsed_s"] >= 0 and entry["finish_reason"] is None and entry["error"] is None
    log.dump(tmp_path / "calls.json")
    assert (tmp_path / "calls.json").is_file()


def test_anthropic_is_optional_and_lazy(monkeypatch):
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert not any("anthropic" in d for d in cfg["project"]["dependencies"])
    assert any(d.startswith("anthropic") for d in cfg["project"]["optional-dependencies"]["llm"])
    assert "anthropic" not in sys.modules  # importing wsp.llm did not import the SDK

    seen = {}

    class Messages:
        def create(self, **kw):
            seen.update(kw)
            return SimpleNamespace(id="msg_1", usage=SimpleNamespace(input_tokens=11, output_tokens=22),
                                   stop_reason="end_turn",
                                   content=[SimpleNamespace(type="text", text="a"),
                                            SimpleNamespace(type="tool_use"),
                                            SimpleNamespace(type="text", text="b")])

    class Anthropic:
        def __init__(self):
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Anthropic))
    c = AnthropicClient()
    assert c.last_usage is None and c.usage_log == []
    assert c.complete("sys", "usr") == "ab"
    assert seen["model"] == "claude-fable-5-1" and seen["system"] == "sys"
    assert seen["messages"] == [{"role": "user", "content": "usr"}] and seen["max_tokens"] == 4096
    assert c.last_usage == {"prompt_tokens": 11, "completion_tokens": 22, "reasoning_tokens": 0, "id": "msg_1",
                            "finish_reason": "end_turn"}
    assert c.usage_log == [c.last_usage]


def test_anthropic_refusal_stop_reason_raises(monkeypatch):
    class Messages:
        def create(self, **kw):
            return SimpleNamespace(id="msg_r", usage=None, stop_reason="refusal", content=[])

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda: SimpleNamespace(messages=Messages())))
    c = AnthropicClient()
    with pytest.raises(LLMRefusal) as ei:
        c.complete("s", "u")
    assert ei.value.finish_reason == "refusal" and ei.value.id == "msg_r"
    assert c.usage_log == [{"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "id": "msg_r",
                            "finish_reason": "refusal"}]


KEY = "test-key-placeholder"


def _ok(content="hi", reasoning=7):
    return 200, json.dumps({"id": "gen-1", "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                                      "completion_tokens_details": {"reasoning_tokens": reasoning}}}).encode()


REFUSAL_BODY = {"id": "gen-refused", "provider": "Anthropic", "model": "anthropic/claude-fable-5.1",
                "object": "chat.completion", "created": 1762214400,
                "choices": [{"logprobs": None, "finish_reason": "content_filter", "native_finish_reason": "refusal",
                             "index": 0, "message": {"role": "assistant", "content": None, "refusal":
                                 "This request triggered restrictions on violative cyber content and was blocked "
                                 "under Anthropic's Usage Policy.", "reasoning": None}}]}


def test_openrouter_refusal_body_raises_llm_refusal():
    c = OpenRouterClient(api_key=KEY, transport=lambda *a: (200, json.dumps(REFUSAL_BODY).encode()))
    with pytest.raises(LLMRefusal) as ei:
        c.complete("s", "u")
    e = ei.value
    assert e.finish_reason == "content_filter" and e.native_finish_reason == "refusal" and e.id == "gen-refused"
    assert e.refusal_text.startswith("This request triggered restrictions") and e.refusal_text in str(e)
    assert c.usage_log == [{"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "id": "gen-refused",
                            "finish_reason": "content_filter"}]
    # The refusal is recorded on the call log entry, then re-raised.
    log = CallLog()
    with pytest.raises(LLMRefusal):
        log.complete(c, "s", "u")
    (entry,) = log.calls
    assert entry["response"] is None and entry["finish_reason"] == "content_filter" and "refusal" in entry["error"]


def test_openrouter_empty_content_without_refusal_is_llm_error():
    body = {"id": "gen-2", "choices": [{"finish_reason": "length", "message": {"content": ""}}]}
    c = OpenRouterClient(api_key=KEY, transport=lambda *a: (200, json.dumps(body).encode()))
    with pytest.raises(LLMError) as ei:
        c.complete("s", "u")
    assert not isinstance(ei.value, LLMRefusal) and ei.value.finish_reason == "length"
    c = OpenRouterClient(api_key=KEY, transport=lambda *a: (200, json.dumps({"id": "gen-3", "choices": []}).encode()))
    with pytest.raises(LLMError) as ei:
        c.complete("s", "u")
    assert not isinstance(ei.value, LLMRefusal) and c.usage_log[-1]["finish_reason"] is None


def _fake_transport(responses, seen):
    def transport(url, headers, body):
        seen.append((url, headers, json.loads(body)))
        return responses.pop(0)
    return transport


def test_openrouter_request_response_and_usage():
    seen = []
    c = OpenRouterClient(api_key=KEY, transport=_fake_transport([_ok()], seen))
    assert c.complete("sys", "usr") == "hi"
    (url, headers, body), = seen
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["Authorization"] == f"Bearer {KEY}" and headers["Content-Type"] == "application/json"
    assert headers["HTTP-Referer"] == "https://github.com/REPLACE-WITH-PUBLIC-REPO" and headers["X-Title"] == "wsp-harness"
    assert body["model"] == "anthropic/claude-fable-5.1" and body["max_tokens"] == 20000
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    assert body["reasoning"] == {"max_tokens": 16000}
    assert c.last_usage == {"prompt_tokens": 10, "completion_tokens": 20, "reasoning_tokens": 7, "id": "gen-1",
                            "finish_reason": "stop"}
    assert c.usage_log == [c.last_usage]
    c2 = OpenRouterClient(api_key=KEY, reasoning_max_tokens=0, model="m", transport=_fake_transport([_ok()], seen))
    c2.complete("s", "u")
    assert "reasoning" not in seen[1][2] and seen[1][2]["model"] == "m"


def test_openrouter_key_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as ei:
        OpenRouterClient(transport=lambda *a: _ok())
    assert str(ei.value) == NO_KEY_MSG == "no OpenRouter key: set OPENROUTER_API_KEY"
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    assert OpenRouterClient(transport=lambda *a: _ok())._key == "env-key"
    assert OpenRouterClient(api_key="arg-key", transport=lambda *a: _ok())._key == "arg-key"


def test_openrouter_error_never_exposes_key():
    def echo_400(url, headers, body):
        return 400, (f"bad request; got {headers['Authorization']}" + "x" * 1000).encode()
    c = OpenRouterClient(api_key=KEY, transport=echo_400)
    with pytest.raises(RuntimeError) as ei:
        c.complete("s", "u")
    assert KEY not in str(ei.value) and "400" in str(ei.value) and len(str(ei.value)) < 600

    def key_across_truncation(url, headers, body):  # the 24-char key spans bytes 480..504, across the 500 cut
        return 400, ("x" * 480 + KEY + "y" * 50).encode()
    c = OpenRouterClient(api_key=KEY, transport=key_across_truncation)
    with pytest.raises(RuntimeError) as ei:
        c.complete("s", "u")
    assert KEY[:8] not in str(ei.value) and "[redacted]" in str(ei.value)

    def raising(url, headers, body):
        raise OSError(f"connection reset while sending {headers['Authorization']}")
    c = OpenRouterClient(api_key=KEY, transport=raising)
    with pytest.raises(RuntimeError) as ei:
        c.complete("s", "u")
    msg = str(ei.value)
    assert KEY not in msg and "transport error" in msg and "OSError" in msg and "[redacted]" in msg
    assert ei.value.__cause__ is None and ei.value.__suppress_context__  # no unsanitized chained exception


def test_openrouter_retries_on_429_not_on_400(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_mod.time, "sleep", sleeps.append)
    seen = []
    c = OpenRouterClient(api_key=KEY, transport=_fake_transport([(429, b"slow"), (503, b"down"), _ok("ok")], seen))
    assert c.complete("s", "u") == "ok" and len(seen) == 3 and sleeps == [2, 4]
    seen.clear()
    c = OpenRouterClient(api_key=KEY, transport=_fake_transport([(400, b"no"), _ok()], seen))
    with pytest.raises(RuntimeError, match="400"):
        c.complete("s", "u")
    assert len(seen) == 1 and sleeps == [2, 4]
    c = OpenRouterClient(api_key=KEY, transport=_fake_transport([(500, b"x")] * 4, seen))
    with pytest.raises(RuntimeError, match="500"):
        c.complete("s", "u")
    assert sleeps == [2, 4, 2, 4, 8]
