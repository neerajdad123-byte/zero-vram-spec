"""Verify-cost curve: ms for one forward pass of K tokens at context depth d.

`llama-bench -p K -n 0 -d d` times exactly a speculative verify pass (K tokens,
one sequence, d tokens already in KV). Run: python -m bench.verify_cost
"""
from __future__ import annotations

import bisect
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

LLAMA_DIR = Path(os.environ.get("STRUCTSPEC_LLAMA_DIR", r"C:\Users\neera\tools\llama.cpp-b11756\bin"))
MODEL = os.environ.get("STRUCTSPEC_MODEL", r"C:\models\gguf\Qwen3.6-35B-A3B-Q4_K_M.gguf")
NCMOE = int(os.environ.get("STRUCTSPEC_NCMOE", "34"))
OUT = Path(__file__).parent / "results" / "verify_cost.json"
BATCHES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
DEPTHS = [512, 2048, 4096]


def measure(reps: int = 3) -> dict:
    cmd = [str(LLAMA_DIR / "llama-bench.exe"), "-m", MODEL, "-ngl", "99", "-ncmoe", str(NCMOE),
           "-p", ",".join(map(str, BATCHES)), "-n", "0", "-d", ",".join(map(str, DEPTHS)),
           "-r", str(reps), "-o", "jsonl"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        rows.append({"batch": r["n_prompt"], "ctx": r["n_depth"],
                     "ms": statistics.median(r["samples_ns"]) / 1e6,
                     "samples_ms": [s / 1e6 for s in r["samples_ns"]]})
    return {"settings": {"cmd": " ".join(cmd[1:]), "ncmoe": NCMOE, "reps": reps}, "rows": rows}


def load_cost(path: Path = OUT):
    """Return cost(batch, ctx) -> ms, linear in ctx, nearest-above measured batch."""
    rows = json.loads(path.read_text())["rows"]
    table: dict[int, list[tuple[int, float]]] = {}
    for r in rows:
        table.setdefault(r["batch"], []).append((r["ctx"], r["ms"]))
    batches = sorted(table)
    for b in batches:
        table[b].sort()

    def cost(batch: int, ctx: int) -> float:
        b = batches[min(bisect.bisect_left(batches, batch), len(batches) - 1)]
        pts = table[b]
        if ctx <= pts[0][0]:
            return pts[0][1]
        for (c0, m0), (c1, m1) in zip(pts, pts[1:]):
            if ctx <= c1:
                return m0 + (m1 - m0) * (ctx - c0) / (c1 - c0)
        (c0, m0), (c1, m1) = pts[-2], pts[-1]
        return m1 + (m1 - m0) * (ctx - c1) / (c1 - c0)

    return cost


if __name__ == "__main__":
    res = measure(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    for r in res["rows"]:
        print(f"batch {r['batch']:>2} ctx {r['ctx']:>5}: {r['ms']:8.1f} ms")
