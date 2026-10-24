"""Predict the first token of a new line from the shape of the previous line.

Features: previous non-blank line's indentation, its last character, its first word
(8 chars), and whether a blank line was just emitted. Learned from code tokenized by the
target model; language-agnostic. Measured on Qwen3.6 tune traces: fires on ~50% of line
starts with ~90% precision. `extend` (e.g. an n-gram model) continues the draft."""
from __future__ import annotations

from collections.abc import Iterable


def line_features(text: str) -> tuple:
    lines = text.split("\n")
    prev = next((ln for ln in reversed(lines[:-1]) if ln.strip()), "")
    indent = prev[:len(prev) - len(prev.lstrip())]
    blank = len(lines) >= 2 and not lines[-2].strip()
    return indent, prev.rstrip()[-1:], prev.strip().split(" ")[0][:8], blank


class LineStartProposer:
    name = "line_start"
    TAIL_TOKENS = 256  # decoded tail must reach back past the previous non-blank line

    def __init__(self, pieces: list[str], corpus: Iterable[list[int]] = (), min_count: int = 2,
                 p_min: float = 0.6, extend=None):
        self.pieces, self.min_count, self.p_min, self.extend = pieces, min_count, p_min, extend
        counts: dict[tuple, dict[int, int]] = {}
        for seq in corpus:
            text = ""
            for tok in seq:
                if text.endswith("\n"):
                    d = counts.setdefault(line_features(text[-4096:]), {})
                    d[tok] = d.get(tok, 0) + 1
                text += pieces[tok]
        self.table = {}
        for f, d in counts.items():
            total = sum(d.values())
            tok, cnt = max(d.items(), key=lambda kv: kv[1])
            if total >= min_count:
                self.table[f] = (tok, cnt / total)

    def propose(self, tokens: list[int], max_k: int) -> list[int]:
        if max_k <= 0 or not tokens:
            return []
        text = "".join(self.pieces[t] for t in tokens[-self.TAIL_TOKENS:])
        if not text.endswith("\n"):
            return []
        hit = self.table.get(line_features(text))
        if not hit or hit[1] < self.p_min:
            return []
        draft = [hit[0]]
        if self.extend is not None and max_k > 1:
            draft += self.extend.propose(list(tokens) + draft, max_k - 1)
        return draft

    def observe(self, accepted: list[int]) -> None:
        if self.extend is not None:
            self.extend.observe(accepted)
