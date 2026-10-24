"""Record greedy traces from a running llama-server (temp 0, thinking off).

Start the server first (settings in docs/research/speed-findings.md), then:
    python -m bench.traces --split test --per-lang 60
Output ids come straight from the server (`return_tokens`), never re-tokenized text,
so offline replay sees exactly what the model emitted. Resumable: done ids are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import httpx

from bench.datasets import load_tasks

RESULTS = Path(__file__).parent / "results" / "traces"
THINK_OPEN = "<think>"
MAX_NEW = {"fresh": 512, "edit": 2048}


def trace_path(split: str) -> Path:
    return RESULTS / f"{split}.jsonl"


def load_traces(split: str) -> list[dict]:
    p = trace_path(split)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def pick(tasks: list[dict], per_lang: int, edit_per_lang: int = 99) -> list[dict]:
    """Stable hash-ordered sample: per_lang fresh and edit_per_lang edit tasks per language."""
    groups: dict[tuple, list[dict]] = {}
    for t in sorted(tasks, key=lambda t: hashlib.sha1(t["id"].encode()).hexdigest()):
        groups.setdefault((t["kind"], t["lang"]), []).append(t)
    cap = {"fresh": per_lang, "edit": edit_per_lang}
    return [t for (kind, _), ts in sorted(groups.items()) for t in ts[:cap[kind]]]


def record(client: httpx.Client, task: dict, think_id: int, max_new: dict = MAX_NEW) -> dict:
    text = client.post("/apply-template", json={"messages": [{"role": "user", "content": task["prompt"]}]}).json()["prompt"]
    prompt_ids = client.post("/tokenize", json={"content": text}).json()["tokens"]
    r = client.post("/completion", json={"prompt": prompt_ids, "n_predict": max_new[task["kind"]], "temperature": 0,
                                         "top_k": 1, "return_tokens": True, "cache_prompt": False}).json()
    out_ids = r["tokens"]
    # content omits the final EOS token (<|im_end|>) that tokens includes
    body = out_ids[:-1] if r.get("stop_type") == "eos" else out_ids
    detok = client.post("/detokenize", json={"tokens": body}).json()["content"]
    t = r["timings"]
    return {"id": task["id"], "lang": task["lang"], "kind": task["kind"], "prompt_ids": prompt_ids,
            "out_ids": out_ids, "n_out": len(out_ids), "decode_ms": t["predicted_ms"], "prompt_ms": t["prompt_ms"],
            "stop_type": r.get("stop_type"), "detok_ok": detok == r["content"], "think_in_out": think_id in out_ids}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["tune", "test"])
    ap.add_argument("--per-lang", type=int, default=60)
    ap.add_argument("--edit-per-lang", type=int, default=99)
    ap.add_argument("--edit-max", type=int, default=MAX_NEW["edit"])
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    a = ap.parse_args()
    out = trace_path(a.split)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {t["id"] for t in load_traces(a.split)}
    todo = [t for t in pick(load_tasks(a.split), a.per_lang, a.edit_per_lang) if t["id"] not in done]
    with httpx.Client(base_url=a.url, timeout=600) as c, out.open("a", encoding="utf-8") as f:
        think_id = c.post("/tokenize", json={"content": THINK_OPEN}).json()["tokens"][0]
        for i, task in enumerate(todo):
            rec = record(c, task, think_id, {"fresh": MAX_NEW["fresh"], "edit": a.edit_max})
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{i + 1}/{len(todo)} {task['id']} n={rec['n_out']} {rec['n_out'] / rec['decode_ms'] * 1e3:.1f} tok/s",
                  flush=True)


if __name__ == "__main__":
    main()
