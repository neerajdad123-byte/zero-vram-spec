"""Copy when the context gives a strong match, otherwise the learned n-gram model."""
from __future__ import annotations

from .ngram_model import NgramModelProposer
from .suffix_copy import SuffixCopyProposer


class ComboProposer:
    name = "combo"

    def __init__(self, copy: SuffixCopyProposer, model: NgramModelProposer, copy_min: int = 6):
        self.copy, self.model, self.copy_min = copy, model, copy_min

    def propose(self, tokens: list[int], max_k: int) -> list[int]:
        if max_k <= 0 or not tokens:
            return []
        c = self.copy.propose(tokens, max_k)
        if c and self.copy.last_match >= self.copy_min:
            return c
        m = self.model.propose(tokens, max_k)
        return m if len(m) >= len(c) else c

    def observe(self, accepted: list[int]) -> None:
        self.copy.observe(accepted)
        self.model.observe(accepted)
