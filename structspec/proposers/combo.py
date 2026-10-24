"""Copy when the context gives a strong match, otherwise the first fallback that drafts
(e.g. context n-gram, then corpus n-gram); a weak copy is used only if it is the longest."""
from __future__ import annotations

from .suffix_copy import SuffixCopyProposer


class ComboProposer:
    name = "combo"

    def __init__(self, copy: SuffixCopyProposer, fallbacks: list, copy_min: int = 6):
        self.copy, self.fallbacks, self.copy_min = copy, list(fallbacks), copy_min

    def propose(self, tokens: list[int], max_k: int) -> list[int]:
        if max_k <= 0 or not tokens:
            return []
        c = self.copy.propose(tokens, max_k)
        if c and self.copy.last_match >= self.copy_min:
            return c
        for f in self.fallbacks:
            m = f.propose(tokens, max_k)
            if m:
                return m if len(m) >= len(c) else c
        return c

    def observe(self, accepted: list[int]) -> None:
        self.copy.observe(accepted)
        for f in self.fallbacks:
            f.observe(accepted)
