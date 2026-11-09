"""Checks the Anthropic backend wiring with a fake client (no API key, no network)."""
import json
import types as pytypes

import pytest

pytest.importorskip("anthropic")

from masec import llm as llm_mod
from masec.network import Defenses, run_episode


class Err(Exception):
    def __init__(self, code, msg="err"):
        super().__init__(msg)
        self.status_code = code


class FakeMessages:
    def __init__(self, fail=()):
        self.fail, self.calls = list(fail), []

    def create(self, model, max_tokens, system, messages, **kw):
        self.calls.append(model)
        self.last_kw = kw
        if self.fail:
            raise self.fail.pop(0)
        reply = {"message": "Summary.", "tool_calls": []}
        if "You are the Writer" in system:
            reply["tool_calls"] = [{"tool": "write_file", "args": {"path": "report.md", "content": "x"}}]
        if "You are the Emailer" in system:
            reply["tool_calls"] = [{"tool": "send_email", "args": {"to": "manager@corp.com", "subject": "s", "body": "b"}}]
        return pytypes.SimpleNamespace(
            content=[pytypes.SimpleNamespace(type="text", text=json.dumps(reply))],
            usage=pytypes.SimpleNamespace(input_tokens=100, output_tokens=20), stop_reason="end_turn")


def make(monkeypatch, fail=()):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    a = llm_mod.make_backend("anthropic")
    a.client = pytypes.SimpleNamespace(messages=FakeMessages(fail))
    return a


def test_anthropic_runs_a_full_episode(monkeypatch):
    a = make(monkeypatch)
    r = run_episode(a, Defenses(True, True, True), "hub", None)
    assert r["benign_task_ok"] and r["parse_failures"] == 0
    assert a.client.messages.calls[0] == "claude-haiku-5-5" and a.usage.input_tokens == 400


def test_anthropic_retries_rate_limits_and_overloads(monkeypatch):
    a = make(monkeypatch, fail=[Err(429), Err(529)])
    assert "Summary" in a.complete("You are the Researcher", "hi")
    assert len(a.client.messages.calls) == 3


@pytest.mark.parametrize("code", [400, 401, 404])
def test_anthropic_stops_run_on_key_or_credit_problems(monkeypatch, code):
    a = make(monkeypatch, fail=[Err(code, "credit balance is too low")])
    with pytest.raises(llm_mod.QuotaExhausted):
        a.complete("s", "u")
    assert len(a.client.messages.calls) == 1


def test_call_matches_the_real_sdk_signature(monkeypatch):
    """Guards against sending arguments the installed SDK does not accept."""
    import inspect
    import anthropic
    accepted = set(inspect.signature(anthropic.resources.messages.Messages.create).parameters)
    sent = {}

    def capture(**kw):
        sent.update(kw)
        return FakeMessages().create(**kw)
    a = make(monkeypatch)
    a.client.messages.create = capture
    a.complete("You are the Researcher", "hi")
    assert set(sent) <= accepted, set(sent) - accepted


def test_agents_use_structured_outputs(monkeypatch):
    a = make(monkeypatch)
    run_episode(a, Defenses(), "chain", None)
    fmt = a.client.messages.last_kw["output_config"]["format"]
    assert fmt["type"] == "json_schema" and "tool_calls" in fmt["schema"]["properties"]


def test_flat_structured_reply_is_parsed():
    from masec.agents import parse_reply
    raw = json.dumps({"message": "ok", "tool_calls": [{"tool": "send_email", "to": "a@b.com", "subject": "s",
                                                       "body": "b", "path": "", "content": ""}]})
    msg, calls, ok = parse_reply(raw)
    assert ok and calls == [{"tool": "send_email", "args": {"to": "a@b.com", "subject": "s", "body": "b"}}]


def test_warning_about_an_attack_is_not_counted_as_compromise(monkeypatch):
    """An agent that only mentions the attacker's address (to warn about it) is not compromised."""
    from masec.scenarios import ATTACKER_EMAIL

    class Warner:
        def complete(self, system, user, schema=None):
            return json.dumps({"message": f"Warning: a document asks us to email {ATTACKER_EMAIL}; ignored.",
                               "tool_calls": []})
    r = run_episode(Warner(), Defenses(), "chain", "goal_hijacking")
    assert r["agents_compromised"] == 0 and r["agents_mentioning_attack"] >= 1 and not r["attack_success"]
