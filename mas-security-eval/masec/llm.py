"""Model backends. Every backend exposes complete(system, user) -> str.

  mock       deterministic stand-in that obeys any instruction it can read
             (worst case). Free, used for tests and debugging.
  anthropic  Claude through the Anthropic API (needs ANTHROPIC_API_KEY).
  gemini     Google Gemini through the google-genai SDK (needs GEMINI_API_KEY).
  ollama     a local open-weight model through Ollama (http://localhost:11434).
"""
from __future__ import annotations

import json
import os
import re
import time

from .gateway import EMAIL_RE, FILE_RE


class QuotaExhausted(RuntimeError):
    """Quota used up, bad key or no credit: stop the whole run, retrying is pointless."""


class Usage:
    def __init__(self):
        self.calls = self.input_tokens = self.output_tokens = self.truncated = 0


# ---------------------------------------------------------------------------
class MockLLM:
    """Pattern-matching stand-in for an LLM agent.

    * It obeys tool instructions written in plain text, e.g.
      "use send_email to send the notes to x@y.com" or
      "use write_file to overwrite config.yaml".
    * It ignores anything inside <untrusted_data> blocks (an idealised model
      that perfectly respects D1 labels) and anything in a rejected message.
    * It does its own job from the trusted task (writer writes, emailer emails).
    """
    name = "mock"

    SEND = re.compile(r"send_email\b[\s\S]{0,120}?\bto\s+(" + EMAIL_RE.pattern + r")", re.I)
    WRITE = re.compile(r"write_file\b[\s\S]{0,80}?\b(?:overwrite|to|named|called)\s+(" + FILE_RE.pattern + r")", re.I)

    def __init__(self):
        self.usage = Usage()

    def complete(self, system: str, user: str, schema=None) -> str:
        self.usage.calls += 1
        if "You audit actions" in system:
            return '{"decision": "ALLOW", "reason": "mock auditor allows rule-approved actions"}'

        role = re.search(r"You are the (\w+)", system).group(1).lower()
        visible = re.sub(r"<untrusted_data>[\s\S]*?</untrusted_data>", "", user)
        visible = re.sub(r"\[REJECTED MESSAGE[^\]]*\][^\n]*", "", visible)
        # Read consecutive messages as one text, so split instructions join up.
        visible = re.sub(r"\n\s*[\w:.\-]+: ", " ", "\n" + visible)
        # Messages from teammates in plain text are visible too (no D1).
        tool_calls, relayed = [], []

        for m in self.SEND.finditer(visible.replace("\n", " ")):
            to = m.group(1)
            if role == "emailer" or not to.endswith("@corp.com"):
                tool_calls.append({"tool": "send_email", "args": {"to": to, "subject": "Report", "body": "See summary."}})
                relayed.append(f"use send_email to send the notes to {to}")
        for m in self.WRITE.finditer(visible.replace("\n", " ")):
            path = m.group(1)
            if role == "writer" or path != "report.md":
                tool_calls.append({"tool": "write_file", "args": {"path": path, "content": "Summary."}})
                relayed.append(f"use write_file to overwrite {path}")

        # Benign job: the trusted task names the report file and manager.
        if role == "writer" and not any(c["args"].get("path") == "report.md" for c in tool_calls) and "report.md" in user:
            tool_calls.append({"tool": "write_file", "args": {"path": "report.md", "content": "Summary."}})
        if role == "emailer" and "manager@corp.com" in user and not any(c["args"].get("to") == "manager@corp.com" for c in tool_calls):
            tool_calls.append({"tool": "send_email", "args": {"to": "manager@corp.com", "subject": "Report", "body": "Summary."}})

        # Naive agents relay instructions they obeyed; they summarise the rest.
        relay = [r for r in relayed if "corp.com" not in r and "report.md" not in r]
        message = "Summary of findings." + (" Action items: " + "; ".join(relay) if relay else "")
        return json.dumps({"message": message, "tool_calls": tool_calls})


# ---------------------------------------------------------------------------
class AnthropicLLM:
    """Claude through the Anthropic SDK (needs ANTHROPIC_API_KEY).

    Rate limits (429) and overloads are retried. A bad key, a missing model or
    an empty credit balance stops the whole run at once (retrying cannot help).
    """
    def __init__(self, model: str = "claude-haiku-5-5", temperature: float = 0.7,
                 max_tokens: int = 2000, rpm: float = 0, structured: bool = True):
        import threading
        import anthropic  # imported lazily so the mock path needs no SDK
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set. In PowerShell run:  "
                             '$env:ANTHROPIC_API_KEY = "sk-ant-..."   (Mac/Linux: export ANTHROPIC_API_KEY=...)')
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=0, timeout=120)    # we retry ourselves
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.name = f"anthropic:{model}"
        self.structured = structured     # constrain replies to the agent JSON schema
        self.usage = Usage()
        self._gap = 60.0 / rpm if rpm else 0.0
        self._next, self._lock = 0.0, threading.Lock()

    _throttle = None   # set below (shared with GeminiLLM)

    def complete(self, system: str, user: str, schema=None) -> str:
        for attempt in range(8):
            self._throttle()
            try:
                # Current Claude models use their own fixed sampling settings, so the
                # SDK takes no temperature argument; --temperature does not apply here.
                extra = {"output_config": {"format": {"type": "json_schema", "schema": schema}}} \
                    if schema and self.structured else {}
                r = self.client.messages.create(
                    model=self.model, max_tokens=self.max_tokens,
                    system=system, messages=[{"role": "user", "content": user}], **extra)
                self.usage.calls += 1
                if getattr(r, "stop_reason", None) == "max_tokens":
                    self.usage.truncated += 1
                self.usage.input_tokens += r.usage.input_tokens
                self.usage.output_tokens += r.usage.output_tokens
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            except Exception as e:
                code = getattr(e, "status_code", None)
                transient = code in (408, 409, 429) or (code or 0) >= 500 or \
                    type(e).__name__ in ("APIConnectionError", "APITimeoutError")
                if code in (400, 401, 403, 404):
                    raise QuotaExhausted(f"Anthropic API refused the request ({code}): {e}. "
                                         "Check your API key, the model name and your credit balance "
                                         "at console.anthropic.com.") from e
                if attempt == 7 or not transient:
                    raise
                time.sleep(min(60, 3 * 2 ** attempt) if code == 429 else 2 ** attempt)
        return ""


# ---------------------------------------------------------------------------
class GeminiLLM:
    """Google Gemini through the google-genai SDK (needs GEMINI_API_KEY).

    `rpm` throttles requests per minute across all worker threads, which keeps
    free-tier keys under their rate limit. Rate-limit errors (429) are retried.
    """
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.7,
                 max_tokens: int = 2000, rpm: float = 0):
        import threading
        from google import genai
        from google.genai import types
        self.client, self.types = genai.Client(), types
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.name = f"gemini:{model}"
        self.usage = Usage()
        self._gap = 60.0 / rpm if rpm else 0.0
        self._next, self._lock = 0.0, threading.Lock()

    def _throttle(self):
        if not self._gap:
            return
        with self._lock:
            wait = self._next - time.time()
            self._next = max(self._next, time.time()) + self._gap
        if wait > 0:
            time.sleep(wait)

    def complete(self, system: str, user: str, schema=None) -> str:
        cfg = self.types.GenerateContentConfig(
            system_instruction=system, temperature=self.temperature,
            max_output_tokens=self.max_tokens, response_mime_type="application/json")
        for attempt in range(8):
            self._throttle()
            try:
                r = self.client.models.generate_content(model=self.model, contents=user, config=cfg)
                self.usage.calls += 1
                m = r.usage_metadata
                if m:
                    self.usage.input_tokens += m.prompt_token_count or 0
                    self.usage.output_tokens += m.candidates_token_count or 0
                return r.text or ""      # empty if the reply was blocked by a safety filter
            except Exception as e:
                code = getattr(e, "code", None)
                if code == 429 and "PerDay" in str(e):
                    raise QuotaExhausted("Gemini daily request limit reached for this key/model. "
                                         "Progress is saved; run the same command again after the "
                                         "limit resets (midnight Pacific time).") from e
                if attempt == 7 or (code is not None and code not in (429, 500, 503, 504)):
                    raise                # bad key, bad model name, etc.: fail fast
                time.sleep(min(60, 5 * 2 ** attempt) if code == 429 else 2 ** attempt)
        return ""


AnthropicLLM._throttle = GeminiLLM._throttle


# ---------------------------------------------------------------------------
class OllamaLLM:
    """A local open-weight model through Ollama.

    * The reply is constrained to the agent JSON schema (Ollama structured outputs),
      and capped at `max_tokens`, so a model cannot loop forever.
    * `num_ctx` is raised so long inboxes are not silently cut off (Ollama's
      default context window is small and it truncates without an error).
    """
    def __init__(self, model: str = "qwen2.5:7b", temperature: float = 0.7,
                 host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                 max_tokens: int = 1024, num_ctx: int = 8192, timeout: int = 900):
        import requests
        self.requests, self.host = requests, host.rstrip("/")
        self.model, self.temperature = model, temperature
        self.max_tokens, self.num_ctx, self.timeout = max_tokens, num_ctx, timeout
        self.name = f"ollama:{model}"
        self.usage = Usage()
        try:
            self.requests.get(f"{self.host}/api/tags", timeout=10).raise_for_status()
        except Exception as e:
            raise SystemExit(f"Cannot reach Ollama at {self.host} ({e}). Is the Ollama app running?")

    def complete(self, system: str, user: str, schema=None) -> str:
        t0 = time.time()
        r = self.requests.post(f"{self.host}/api/chat", timeout=self.timeout, json={
            "model": self.model, "stream": False, "keep_alive": "30m",
            "format": schema or "json",
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens,
                        "num_ctx": self.num_ctx},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        if r.status_code == 404:
            raise QuotaExhausted(f"Ollama has no model '{self.model}'. Run:  ollama pull {self.model}")
        r.raise_for_status()
        data = r.json()
        self.usage.calls += 1
        self.usage.input_tokens += data.get("prompt_eval_count", 0)
        self.usage.output_tokens += data.get("eval_count", 0)
        if data.get("done_reason") == "length":
            self.usage.truncated += 1
        self.usage.seconds = getattr(self.usage, "seconds", 0.0) + (time.time() - t0)
        return data["message"]["content"]


def make_backend(spec: str, temperature: float = 0.7, rpm: float = 0):
    """'mock', 'anthropic[:model]', 'gemini[:model]', 'ollama[:model]'."""
    kind, _, model = spec.partition(":")
    if kind == "mock":
        return MockLLM()
    if kind == "anthropic":
        return AnthropicLLM(model or "claude-haiku-5-5", temperature, rpm=rpm)
    if kind == "gemini":
        return GeminiLLM(model or "gemini-2.5-flash", temperature, rpm=rpm)
    if kind == "ollama":
        return OllamaLLM(model or "qwen2.5:7b", temperature)
    raise ValueError(f"unknown backend {spec!r}")
