"""Exact offline replay of proposers against greedy traces.

Under greedy verification a draft token is accepted iff it equals the greedy token,
so replaying a proposer against a recorded greedy trace gives its exact acceptance.
Speedup is predicted from the measured verify-cost curve (bench/verify_cost.py).

    python -m bench.replay --split tune --proposers prompt_copy,bracket_close --max-k 8
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path


def replay(proposer, trace: dict, max_k: int, cost=None) -> dict:
    prompt, out = trace["prompt_ids"], trace["out_ids"]
    n = len(out)
    passes = accepted = drafted = with_draft = 0
    prop_s = spec_ms = greedy_ms = 0.0
    i = 0
    while i < n:
        ctx = prompt + out[:i]
        t0 = time.perf_counter()
        draft = list(proposer.propose(ctx, max_k))[:max_k]
        prop_s += time.perf_counter() - t0
        a = 0
        while a < len(draft) and i + a < n and draft[a] == out[i + a]:
            a += 1
        step = min(a + 1, n - i)  # the verify pass also yields the model's own next token
        proposer.observe(out[i:i + step])
        if cost:
            spec_ms += cost(len(draft) + 1, len(ctx))
            greedy_ms += sum(cost(1, len(ctx) + j) for j in range(step))
        passes += 1
        accepted += a
        drafted += len(draft)
        with_draft += bool(draft)
        i += step
    return {"passes": passes, "accepted": accepted, "drafted": drafted, "steps_with_draft": with_draft,
            "proposer_s": prop_s, "n_out": n, "spec_ms": spec_ms, "greedy_ms": greedy_ms}


def predict(r: dict) -> float:
    """Predicted speedup = greedy_time / (sum of verify-pass costs + proposer time)."""
    return r["greedy_ms"] / (r["spec_ms"] + r["proposer_s"] * 1e3)


def summarize(rows: list[dict]) -> dict:
    s = {k: sum(r[k] for r in rows) for k in rows[0] if isinstance(rows[0][k], (int, float))}
    return {"tasks": len(rows), "tokens": s["n_out"],
            "coverage": s["steps_with_draft"] / s["passes"],
            "acceptance": s["accepted"] / s["drafted"] if s["drafted"] else 0.0,
            "mean_accept_when_drafted": s["accepted"] / s["steps_with_draft"] if s["steps_with_draft"] else 0.0,
            "tokens_per_pass": s["n_out"] / s["passes"],
            "pred_speedup": predict(s) if s["greedy_ms"] else None,
            "proposer_us_per_tok": s["proposer_s"] * 1e6 / s["n_out"]}


def run(make, traces: list[dict], max_k: int, cost) -> dict:
    """make(trace) -> fresh proposer per task. Returns summaries keyed by (lang, kind), lang, kind, 'all'."""
    groups: dict = defaultdict(list)
    for t in traces:
        r = replay(make(t), t, max_k, cost)
        for key in [(t["lang"], t["kind"]), t["lang"], t["kind"], "all"]:
            groups[key].append(r)
    return {k: summarize(v) for k, v in groups.items()}


def fmt_table(res: dict, name: str) -> str:
    keys = ["all", "fresh", "edit"] + sorted(k for k in res if isinstance(k, str) and k not in ("all", "fresh", "edit"))
    lines = [f"### {name}", "| group | tasks | tokens | coverage | accept rate | mean acc len | tok/pass | pred speedup |",
             "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        if k in res:
            s = res[k]
            lines.append(f"| {k} | {s['tasks']} | {s['tokens']} | {s['coverage']:.1%} | {s['acceptance']:.1%} | "
                         f"{s['mean_accept_when_drafted']:.2f} | {s['tokens_per_pass']:.2f} | {s['pred_speedup']:.2f}x |")
    return "\n".join(lines)


def main() -> None:
    from bench.candidates import REGISTRY
    from bench.traces import load_traces
    from bench.verify_cost import load_cost

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["tune", "test"])
    ap.add_argument("--proposers", required=True)
    ap.add_argument("--max-k", type=int, default=8)
    ap.add_argument("--save")
    a = ap.parse_args()
    traces, cost = load_traces(a.split), load_cost()
    saved = {}
    for name in a.proposers.split(","):
        res = run(lambda t: REGISTRY[name](t), traces, a.max_k, cost)
        print(fmt_table(res, f"{name} (K={a.max_k}, {a.split})"), "\n")
        saved[name] = {"|".join(k) if isinstance(k, tuple) else k: v for k, v in res.items()}
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(json.dumps({"split": a.split, "max_k": a.max_k, "results": saved}, indent=1))


if __name__ == "__main__":
    main()
