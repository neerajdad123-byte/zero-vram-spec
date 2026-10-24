"""Honest bar: run llama-server's own speculation on test tasks whose greedy trace we have.

Start llama-server with the --spec-type under test, then:
    python -m bench.server_bench --label draft-mtp --per-lang 2
Reports tok/s, llama.cpp's draft acceptance, and whether output == the greedy trace.
Appends to bench/results/server_bench.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from bench.traces import load_traces

OUT = Path(__file__).parent / "results" / "server_bench.jsonl"


def pick_traces(per_lang: int) -> list[dict]:
    seen: dict[str, int] = {}
    out = []
    for t in sorted(load_traces("test"), key=lambda t: t["id"]):
        if t["kind"] == "fresh" and seen.get(t["lang"], 0) < per_lang:
            seen[t["lang"]] = seen.get(t["lang"], 0) + 1
            out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--per-lang", type=int, default=2)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    a = ap.parse_args()
    with httpx.Client(base_url=a.url, timeout=600) as c, OUT.open("a", encoding="utf-8") as f:
        for t in pick_traces(a.per_lang):
            r = c.post("/completion", json={"prompt": t["prompt_ids"], "n_predict": len(t["out_ids"]),
                                            "temperature": 0, "top_k": 1, "return_tokens": True,
                                            "cache_prompt": False}).json()
            tm = r["timings"]
            rec = {"label": a.label, "id": t["id"], "lang": t["lang"], "n": len(r["tokens"]),
                   "tok_s": len(r["tokens"]) / tm["predicted_ms"] * 1e3,
                   "greedy_tok_s": t["n_out"] / t["decode_ms"] * 1e3,
                   "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted"),
                   "identical": r["tokens"] == t["out_ids"]}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(rec, flush=True)


if __name__ == "__main__":
    main()
