"""Longest-suffix copy: find where the current token suffix occurred before (prompt or
output) and draft what followed it there. Covers prompt copy, edits and repeated code."""
from __future__ import annotations


class SuffixCopyProposer:
    name = "suffix_copy"

    def __init__(self, min_match: int = 3, max_match: int = 64, max_candidates: int = 32,
                 len_slope: float = 1.0, len_base: int = 1):
        """Draft length = len_base + len_slope * (match_len - min_match), capped by max_k:
        longer matches earn longer drafts."""
        self.min_match, self.max_match, self.max_candidates = min_match, max_match, max_candidates
        self.len_slope, self.len_base = len_slope, len_base
        self._toks: list[int] = []
        self._index: dict[tuple, list[int]] = {}
        self.last_match = 0

    def _sync(self, tokens: list[int]) -> None:
        n0 = len(self._toks)
        if n0 > len(tokens) or (n0 and tokens[n0 - 1] != self._toks[-1]):
            self._toks, self._index, n0 = [], {}, 0
        m = self.min_match
        for p in range(max(n0, m), len(tokens) + 1):
            self._index.setdefault(tuple(tokens[p - m:p]), []).append(p)
        self._toks = list(tokens)

    def match(self, tokens: list[int]) -> tuple[int, int]:
        """(start of continuation, match length) of the best earlier occurrence, or (-1, 0)."""
        self._sync(tokens)
        n = len(tokens)
        if n < self.min_match + 1:
            return -1, 0
        best_p, best_len = -1, 0
        cands = self._index.get(tuple(tokens[n - self.min_match:]), [])
        for p in reversed(cands[-self.max_candidates - 1:]):
            if p >= n:
                continue
            ln = self.min_match
            while ln < self.max_match and p - 1 - ln >= 0 and tokens[p - 1 - ln] == tokens[n - 1 - ln]:
                ln += 1
            if ln > best_len:
                best_p, best_len = p, ln
        return best_p, best_len

    def propose(self, tokens: list[int], max_k: int) -> list[int]:
        self.last_match = 0
        if max_k <= 0 or not tokens:
            return []
        p, ln = self.match(tokens)
        if p < 0:
            return []
        self.last_match = ln
        k = min(max_k, max(1, int(self.len_base + self.len_slope * (ln - self.min_match))))
        return list(tokens[p:p + k])

    def observe(self, accepted: list[int]) -> None:
        pass  # state is rebuilt from the tokens passed to propose
