from __future__ import annotations

import json
import threading

import pytest
from fake_model import FakeModel, greedy_continuation
from fastapi.testclient import TestClient

from structspec.llama_backend import chat_prompt_ids, run_greedy
from structspec.proposer import PromptLookupProposer
from structspec.server import create_spec_app


class Oracle:
    name = "oracle"

    def propose(self, tokens, max_k):
        return greedy_continuation(tokens, max_k)

    def observe(self, accepted):
        pass


def _client(n_ctx=4096, proposer=None):
    model = FakeModel(n_ctx=n_ctx)
    return TestClient(create_spec_app(model, proposer or Oracle(), k=4, model_name="fake")), model


def _greedy_text(prompt_ids, max_tokens):
    model = FakeModel()
    ids = run_greedy(model, prompt_ids, max_tokens)[0][len(prompt_ids) :]
    return model.decode([t for t in ids if t != model.eos]), ids


def _sse(resp):
    events = [line[len("data: ") :] for line in resp.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    return [json.loads(e) for e in events[:-1]]


def _find_prompt(pred, max_tokens=48):
    """A prompt whose greedy output satisfies pred(text, ids)."""
    model = FakeModel()
    for i in range(500):
        prompt = f"x{i} = foo("
        text, ids = _greedy_text(model.encode(prompt), max_tokens)
        if pred(text, ids):
            return prompt, text
    pytest.fail("no suitable prompt")


def test_completion_equals_greedy():
    client, model = _client()
    r = client.post("/v1/completions", json={"prompt": "def add(a, b):", "max_tokens": 40})
    assert r.status_code == 200
    body = r.json()
    want, ids = _greedy_text(model.encode("def add(a, b):"), 40)
    assert body["choices"][0]["text"] == want
    assert body["usage"]["completion_tokens"] == len(ids)
    assert body["choices"][0]["finish_reason"] == ("stop" if ids[-1] == model.eos else "length")


def test_chat_completion_equals_greedy():
    client, model = _client()
    msgs = [{"role": "user", "content": "hello there"}]
    r = client.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 30})
    assert r.status_code == 200
    msg = r.json()["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert msg["content"] == _greedy_text(chat_prompt_ids(model, msgs), 30)[0]


def test_stream_concat_equals_nonstream_with_split_utf8():
    prompt, want = _find_prompt(lambda text, ids: "é" in text)
    client, _ = _client()
    r = client.post("/v1/completions", json={"prompt": prompt, "max_tokens": 48, "stream": True})
    chunks = _sse(r)
    streamed = "".join(c["choices"][0]["text"] for c in chunks)
    assert streamed == want and "�" not in streamed
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")


def test_chat_stream_matches_nonstream():
    client, _ = _client()
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 30}
    full = client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]["content"]
    chunks = _sse(client.post("/v1/chat/completions", json={**body, "stream": True}))
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == full


def test_stop_string_truncates():
    prompt, text = _find_prompt(lambda text, ids: len(text) > 12 and text[6:9].strip() == text[6:9])
    stop = text[6:9]
    cut = text[: text.index(stop)]
    client, _ = _client()
    body = {"prompt": prompt, "max_tokens": 48, "stop": [stop]}
    r = client.post("/v1/completions", json=body).json()
    assert r["choices"][0]["text"] == cut and r["choices"][0]["finish_reason"] == "stop"
    streamed = "".join(c["choices"][0]["text"] for c in _sse(client.post("/v1/completions", json={**body, "stream": True})))
    assert streamed == cut


def test_prompt_too_long_returns_400():
    client, _ = _client(n_ctx=16)
    r = client.post("/v1/completions", json={"prompt": "a" * 40, "max_tokens": 4})
    assert r.status_code == 400
    assert "context" in r.json()["error"]["message"]


def test_max_tokens_clamped_to_context():
    client, _ = _client(n_ctx=32)
    r = client.post("/v1/completions", json={"prompt": "a" * 20, "max_tokens": 1000})
    assert r.status_code == 200
    assert r.json()["usage"]["completion_tokens"] <= 12


def test_bad_request_returns_400():
    client, _ = _client()
    assert client.post("/v1/completions", json={"max_tokens": 4}).status_code == 400
    assert client.post("/v1/chat/completions", json={"messages": "nope"}).status_code == 400
    assert client.post("/v1/completions", json={"prompt": "a", "max_tokens": 0}).status_code == 400


def test_concurrent_requests_serialized():
    client, model = _client(proposer=PromptLookupProposer())
    prompts = ["def f(a):", "class A:"]
    results = {}

    def go(p):
        results[p] = client.post("/v1/completions", json={"prompt": p, "max_tokens": 40}).json()["choices"][0]["text"]

    threads = [threading.Thread(target=go, args=(p,)) for p in prompts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for p in prompts:
        assert results[p] == _greedy_text(model.encode(p), 40)[0]


def test_metrics_reflect_real_counts():
    client, model = _client()
    r = client.post("/v1/completions", json={"prompt": "def add(a, b):", "max_tokens": 40}).json()
    health = client.get("/health").json()
    assert health["decoding"] == "greedy" and health["mode"] == "speculative"
    assert health["generated_tokens"] == r["usage"]["completion_tokens"]
    assert 1 <= health["target_passes"] <= health["generated_tokens"]
    assert health["accepted_draft_tokens"] <= health["proposed_draft_tokens"]
    assert health["requests_total"] == 1
    assert "structspec_generated_tokens" in client.get("/v1/structspec/metrics").text


def test_models_endpoint():
    client, _ = _client()
    assert client.get("/v1/models").json()["data"][0]["id"] == "fake"
