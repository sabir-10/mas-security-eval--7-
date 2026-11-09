"""Checks the Ollama backend wiring with a fake HTTP layer (no Ollama install needed)."""
import json
import types as pytypes

from masec.llm import make_backend
from masec.network import Defenses, run_episode


class FakeRequests:
    def __init__(self):
        self.sent = []

    def get(self, url, timeout):
        return pytypes.SimpleNamespace(raise_for_status=lambda: None)

    def post(self, url, timeout, json):
        self.sent.append((url, json))
        system = json["messages"][0]["content"]
        reply = {"message": "Summary.", "tool_calls": []}
        if "You are the Writer" in system:
            reply["tool_calls"] = [{"tool": "write_file", "args": {"path": "report.md", "content": "x"}}]
        if "You are the Emailer" in system:
            reply["tool_calls"] = [{"tool": "send_email", "args": {"to": "manager@corp.com", "subject": "s", "body": "b"}}]
        body = {"message": {"content": __import__("json").dumps(reply)}, "prompt_eval_count": 50, "eval_count": 10}
        return pytypes.SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: body)


def make(monkeypatch):
    import requests
    fake = FakeRequests()
    monkeypatch.setattr(requests, "get", fake.get)
    o = make_backend("ollama:qwen2.5:7b")
    o.requests = fake
    return o


def test_ollama_runs_a_full_episode(monkeypatch):
    o = make(monkeypatch)
    r = run_episode(o, Defenses(True, True, True), "chain", None)
    assert r["benign_task_ok"] and r["parse_failures"] == 0
    url, payload = o.requests.sent[0]
    assert url.endswith("/api/chat") and payload["model"] == "qwen2.5:7b"
    assert payload["format"]["type"] == "object"                 # structured output schema
    assert payload["options"]["num_predict"] == 1024 and payload["options"]["num_ctx"] == 8192
    assert o.usage.calls == 4
