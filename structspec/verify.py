"""Greedy-vs-speculative identity check on a real model, with mismatch root-causing."""

from __future__ import annotations

import numpy as np

from .llama_backend import run_greedy, run_speculative

# Top-1 minus top-2 logit below this counts as a near tie that batch-shape numerics can flip.
NUMERICS_MARGIN = 0.5

DEFAULT_PROMPTS = [
    "Write a Python function that parses an ISO 8601 date string and returns a datetime. Code only.",
    "Write a TypeScript interface for a blog post with author, tags and timestamps, "
    "plus a function that sorts posts by date. Code only.",
    "Convert this to async/await:\nfunction load(url, cb) { fetch(url).then(r => r.json())"
    ".then(d => cb(null, d)).catch(e => cb(e)); }",
    'Write a Go HTTP handler that returns JSON {"status": "ok"}. Code only.',
    "Add type hints to this Python code:\ndef merge(a, b):\n    out = dict(a)\n    for k, v in b.items():\n"
    "        out[k] = v\n    return out",
    "Write a SQL query that returns the top 5 customers by total order value from tables "
    "customers(id, name) and orders(id, customer_id, total).",
    "Write a Rust function that reads a file line by line and counts non-empty lines. Code only.",
    "Write a bash script that backs up a directory into a timestamped tar.gz. Code only.",
]


def _margin(row: np.ndarray) -> float:
    top2 = np.partition(row, -2)[-2:]
    return float(top2[1] - top2[0])


def _replay_row(model, prompt_ids: list[int], gen: list[int], pos: int, batch: int) -> np.ndarray:
    """Logits predicting gen[pos], computed like greedy (prompt prefill, then single tokens), except the
    last `batch` inputs go through one batched decode."""
    model.reset()
    seq = gen[:pos]
    split = max(0, len(seq) - batch) if batch > 1 else len(seq) - 1 if seq else 0
    row = model.decode_logits(prompt_ids, logits_all=False)[-1]
    for tok in seq[:split]:
        row = model.decode_logits([tok], logits_all=False)[-1]
    if seq[split:]:
        row = model.decode_logits(seq[split:], logits_all=batch > 1)[-1]
    return row


def verify_prompt(model, proposer, prompt_ids: list[int], max_tokens: int, k: int = 4) -> dict:
    g = run_greedy(model, prompt_ids, max_tokens)[0][len(prompt_ids) :]
    s, _, stats = run_speculative(model, proposer, prompt_ids, max_tokens, k=k)
    s = s[len(prompt_ids) :]
    first = next((i for i, (a, b) in enumerate(zip(g, s)) if a != b), None)
    if first is None and len(g) != len(s):
        first = min(len(g), len(s))
    result = {
        "identical": first is None,
        "first_mismatch": first,
        "tokens": len(g),
        "accepted": stats["accepted"],
        "proposed": stats["proposed"],
        "greedy_margin": None,
        "batched_argmax": None,
        "classification": "identical",
    }
    if first is None:
        return result
    single = _replay_row(model, prompt_ids, g, first, 1)
    batched = _replay_row(model, prompt_ids, g, first, k + 1)
    result["greedy_margin"] = _margin(single)
    result["batched_argmax"] = int(np.argmax(batched))
    explained = first < len(s) and result["batched_argmax"] == s[first] != int(np.argmax(single))
    result["classification"] = "numerics" if explained and result["greedy_margin"] < NUMERICS_MARGIN else "bug"
    return result
