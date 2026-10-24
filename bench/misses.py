"""Which kinds of tokens does a proposer miss? Classifies every greedy output token and
reports coverage (accepted via a draft) per class, on the TUNE split.

    python -m bench.misses combo
"""
from __future__ import annotations

import re
import sys
from collections import Counter

from bench.candidates import REGISTRY
from bench.corpus import vocab_pieces
from bench.traces import load_traces

CLOSERS = set(")]}>;,\"'`")


def classify(piece: str, prev: str, ctx_text: str) -> str:
    if piece == "\n" or set(piece) <= {"\n"}:
        return "newline"
    if piece.strip() == "":
        return "indent" if prev.endswith("\n") else "space"
    s = piece.strip()
    if all(c in CLOSERS for c in s):
        return "closer"
    if all(not c.isalnum() and c != "_" for c in s):
        return "punct/op"
    word = re.sub(r"\W", "", s)
    if word and re.search(r"(?<!\w)" + re.escape(word), ctx_text):
        return "word-in-context"
    return "new-word"


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "combo"
    pieces = vocab_pieces()
    tot, hit = Counter(), Counter()
    for t in load_traces("tune"):
        p = REGISTRY[name](t)
        prompt, out = t["prompt_ids"], t["out_ids"]
        text = "".join(pieces[x] for x in prompt)
        i = 0
        covered = set()
        while i < len(out):
            d = p.propose(prompt + out[:i], 8)
            a = 0
            while a < len(d) and i + a < len(out) and d[a] == out[i + a]:
                a += 1
            covered.update(range(i, i + a))
            step = min(a + 1, len(out) - i)
            p.observe(out[i:i + step])
            i += step
        prev = text
        for j, tok in enumerate(out):
            c = (t["kind"], classify(pieces[tok], prev[-1:] if prev else "", prev))
            tot[c] += 1
            hit[c] += j in covered
            prev += pieces[tok]
    n = sum(tot.values())
    print(f"| kind | class | share of tokens | covered by {name} | uncovered share of all |")
    print("|---|---|---|---|---|")
    for c, v in sorted(tot.items(), key=lambda kv: -(kv[1] - hit[kv[0]])):
        print(f"| {c[0]} | {c[1]} | {v / n:.1%} | {hit[c] / v:.1%} | {(v - hit[c]) / n:.1%} |")


if __name__ == "__main__":
    main()
