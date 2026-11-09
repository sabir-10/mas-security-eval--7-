"""Checks the Gemini backend wiring with a fake client (no API key, no network)."""
import json
import types as pytypes

import pytest

genai = pytest.importorskip("google.genai")

from masec import llm as llm_mod
from masec.network import Defenses, run_episode


class FakeModels:
    def __init__(self, fail_first=0):
        self.calls, self.fail_first = [], fail_first

    def generate_content(self, model, contents, config):
        self.calls.append((model, config))
        if self.fail_first:
            self.fail_first -= 1
            err = Exception("rate limited")
            err.code = 429
            raise err
        reply = {"message": "Summary of findings.", "tool_calls": []}
        if "You are the Writer" in config.system_instruction:
            reply["tool_calls"] = [{"tool": "write_file", "args": {"path": "report.md", "content": "x"}}]
        if "You are the Emailer" in config.system_instruction:
            reply["tool_calls"] = [{"tool": "send_email", "args": {"to": "manager@corp.com",
                                                                  "subject": "s", "body": "b"}}]
        return pytypes.SimpleNamespace(text=json.dumps(reply), usage_metadata=pytypes.SimpleNamespace(
            prompt_token_count=100, candidates_token_count=20))


def make(monkeypatch, **kw):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    g = llm_mod.make_backend("gemini:gemini-test", 0.5, **kw)
    g.client = pytypes.SimpleNamespace(models=FakeModels(kw.pop("fail_first", 0)))
    return g


def test_gemini_runs_a_full_episode(monkeypatch):
    g = make(monkeypatch)
    r = run_episode(g, Defenses(True, True, True), "hub", None)
    assert r["benign_task_ok"] and r["parse_failures"] == 0
    model, cfg = g.client.models.calls[0]
    assert model == "gemini-test" and cfg.response_mime_type == "application/json"
    assert g.usage.calls == 4 and g.usage.input_tokens == 400


def test_gemini_retries_rate_limits(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    g = make(monkeypatch)
    g.client.models.fail_first = 2
    assert "Summary" in g.complete("You are the Researcher", "hi")
    assert len(g.client.models.calls) == 3


def test_gemini_fails_fast_on_bad_request(monkeypatch):
    g = make(monkeypatch)

    def bad(**kw):
        e = Exception("model not found")
        e.code = 404
        raise e
    g.client.models.generate_content = bad
    with pytest.raises(Exception, match="model not found"):
        g.complete("s", "u")


def test_gemini_daily_quota_stops_immediately(monkeypatch):
    g = make(monkeypatch)
    calls = []

    def quota(**kw):
        calls.append(1)
        e = Exception("429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        e.code = 429
        raise e
    g.client.models.generate_content = quota
    with pytest.raises(llm_mod.QuotaExhausted):
        g.complete("s", "u")
    assert len(calls) == 1          # no pointless retries
