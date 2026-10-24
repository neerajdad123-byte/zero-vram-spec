import pytest

from bench.traces import load_traces, pick

TRACES = load_traces("test") + load_traces("tune")
needs_traces = pytest.mark.skipif(not TRACES, reason="run python -m bench.traces first")


def test_pick_keeps_edits_and_caps_fresh():
    tasks = [{"id": f"f{i}", "lang": "py", "kind": "fresh"} for i in range(10)] + [
        {"id": "e1", "lang": "go", "kind": "edit"}]
    got = pick(tasks, 3)
    assert sum(t["kind"] == "fresh" for t in got) == 3 and any(t["id"] == "e1" for t in got)
    assert [t["id"] for t in pick(tasks, 3)] == [t["id"] for t in got]


@needs_traces
def test_trace_tokens_match_completion():
    assert all(t["detok_ok"] for t in TRACES)


@needs_traces
def test_no_think_tokens():
    assert not any(t["think_in_out"] for t in TRACES)
