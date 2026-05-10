from __future__ import annotations

import time

from structspec.metrics import Metrics, MetricsStore


def test_metrics_defaults():
    m = Metrics()
    assert m.requests_total == 0
    assert m.acceptance_rate == 0.0
    assert m.estimated_multiplier == 1.0
    assert m.uptime_s >= 0.0


def test_metrics_acceptance_rate():
    m = Metrics()
    m.proposed_draft_tokens = 100
    m.accepted_draft_tokens = 75
    assert m.acceptance_rate == 0.75


def test_metrics_multiplier():
    m = Metrics()
    m.target_tokens = 100
    m.accepted_draft_tokens = 50
    assert m.estimated_multiplier == 1.5


def test_metrics_store_snapshot():
    store = MetricsStore()
    store.inc("requests_total", 5)
    snap = store.snapshot()
    assert snap["requests_total"] == 5
    assert "uptime_s" in snap


def test_metrics_store_set_backend():
    store = MetricsStore()
    store.set_backend("vllm", "Qwen-7B")
    snap = store.snapshot()
    assert snap["last_backend"] == "vllm"
    assert snap["last_model"] == "Qwen-7B"


def test_metrics_prometheus_output():
    store = MetricsStore()
    store.inc("requests_total", 3)
    prom = store.prometheus()
    lines = prom.strip().splitlines()
    assert any("structspec_requests_total 3" in line for line in lines)
    assert any("structspec_uptime_s" in line for line in lines)
