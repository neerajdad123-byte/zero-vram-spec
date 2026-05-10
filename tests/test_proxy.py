from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from structspec.proxy import create_app, infer_target_base_url


def test_infer_target_base_url_defaults():
    assert infer_target_base_url("vllm") == "http://localhost:8000/v1"
    assert infer_target_base_url("ollama") == "http://localhost:11434/v1"
    assert infer_target_base_url("lm-studio") == "http://localhost:1234/v1"

    explicit = "http://example.com/v1"
    assert infer_target_base_url("vllm", explicit) == "http://example.com/v1"


def test_health_endpoint():
    app = create_app("http://localhost:8000/v1", backend="vllm")
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["backend"] == "vllm"
    assert "domains" in data
    assert "python" in data["domains"]


def test_root_endpoint():
    app = create_app("http://localhost:8000/v1")
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["name"] == "structspec"


def test_prometheus_endpoint():
    app = create_app("http://localhost:8000/v1")
    client = TestClient(app)
    resp = client.get("/v1/structspec/metrics")
    assert resp.status_code == 200
    assert "structspec_requests_total" in resp.text
