"""End-to-end wall-clock: plain greedy vs speculative decoding, same in-process model.

For each held-out test task: run greedy, then each proposer, and record wall time and
whether the speculative output is byte-identical to greedy. Stops at --budget seconds.

    python -m bench.e2e --model C:/models/gguf/qwen2.5-coder-3b-instruct-q4_k_m.gguf --budget 600
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from bench.datasets import load_tasks
from structspec.llama_backend import chat_prompt_ids, run_greedy, run_speculative
from structspec.llama_native import NativeLlama
from structspec.proposer import PromptLookupProposer
from structspec.proposers import ComboProposer, ContextNgramProposer, SuffixCopyProposer

PROPOSERS = {
    "prompt_lookup": lambda: PromptLookupProposer(),  # previous default
    "suffix_copy": lambda: SuffixCopyProposer(min_match=2, len_slope=0.5, len_base=1),
    "copy+context": lambda: ComboProposer(SuffixCopyProposer(min_match=2, len_slope=0.5, len_base=1),
                                          [ContextNgramProposer(min_order=2, min_count=1)], copy_min=2),
}
MAX_NEW = {"fresh": 384, "edit": 768}


def interleaved(tasks: list[dict]) -> list[dict]:
    """Round-robin over (lang, kind) groups so a time budget still covers every group."""
    groups: dict = defaultdict(list)
    for t in sorted(tasks, key=lambda t: hashlib.sha1(t["id"].encode()).hexdigest()):
        groups[(t["lang"], t["kind"])].append(t)
    out, i = [], 0
    while any(len(g) > i for g in groups.values()):
        out += [g[i] for _, g in sorted(groups.items()) if len(g) > i]
        i += 1
    return out


def summarize(rows: list[dict], names: list[str]) -> str:
    def line(label, rs):
        g = sum(r["greedy_s"] for r in rs)
        n = sum(r["n"] for r in rs)
        cells = []
        for p in names:
            s = sum(r[p]["s"] for r in rs)
            acc = sum(r[p]["accepted"] for r in rs) / max(1, sum(r[p]["proposed"] for r in rs))
            med = statistics.median(r["greedy_s"] / r[p]["s"] for r in rs)
            cells.append(f"{g / s:.2f}x ({med:.2f}x med, {acc:.0%} acc)")
        return f"| {label} | {len(rs)} | {n} | {n / g:.1f} | " + " | ".join(cells) + " |"

    head = ("| group | tasks | tokens | greedy tok/s | " + " | ".join(f"{p} speedup" for p in names) + " |\n"
            "|---|---|---|---|" + "---|" * len(names))
    lines = [head, line("all", rows)]
    for key in ["fresh", "edit"]:
        rs = [r for r in rows if r["kind"] == key]
        if rs:
            lines.append(line(key, rs))
    for lang in ["py", "js", "ts", "go", "rs", "java", "cpp"]:
        rs = [r for r in rows if r["lang"] == lang]
        if rs:
            lines.append(line(lang, rs))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=float, default=600)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--ngl", type=int, default=-1)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "e2e.jsonl"))
    a = ap.parse_args()
    model = NativeLlama(a.model, n_ctx=8192, n_gpu_layers=a.ngl, max_rollback=a.k + 1)
    names = list(PROPOSERS)
    rows = []
    warm = chat_prompt_ids(model, [{"role": "user", "content": "Write hello world in C."}])
    run_greedy(model, warm, 16)  # load weights to GPU before timing
    t_end = time.perf_counter() + a.budget
    with open(a.out, "w", encoding="utf-8") as f:
        for task in interleaved(load_tasks("test")):
            if time.perf_counter() > t_end:
                break
            ids = chat_prompt_ids(model, [{"role": "user", "content": task["prompt"]}])
            if len(ids) + MAX_NEW[task["kind"]] > 8192:
                continue
            g_ids, _, g = run_greedy(model, ids, MAX_NEW[task["kind"]])
            row = {"id": task["id"], "lang": task["lang"], "kind": task["kind"], "n": g["generated"],
                   "greedy_s": g["time"]}
            for name in names:
                s_ids, _, s = run_speculative(model, PROPOSERS[name](), ids, MAX_NEW[task["kind"]], k=a.k)
                row[name] = {"s": s["time"], "accepted": s["accepted"], "proposed": s["proposed"],
                             "passes": s["passes"], "identical": s_ids == g_ids}
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(task["id"], row["n"], " ".join(f"{n}={row['greedy_s'] / row[n]['s']:.2f}x" for n in names),
                  flush=True)
    print(summarize(rows, names))
    print("identical:", {n: sum(r[n]["identical"] for r in rows) for n in names}, "of", len(rows))


if __name__ == "__main__":
    main()
