"""Backoff n-gram model over token ids, trained on code tokenized by the target model.

Drafts greedily from the longest context seen at least `min_count` times and stops when
the chained probability drops below `cum_min` (the verify-cost break-even, ~0.18 on
Qwen3.6-35B-A3B). Learns boilerplate (closing braces, `if err != nil {`, fences) from data
instead of hand-written rules."""
from __future__ import annotations

from collections.abc import Iterable


class NgramModelProposer:
    name = "ngram_model"

    def __init__(self, corpus: Iterable[list[int]] = (), max_order: int = 6, min_order: int = 2,
                 min_count: int = 2, p_min: float = 0.5, cum_min: float = 0.2, alpha: float = 0.0):
        """alpha: additive penalty on the context total, P = count / (total + alpha); raw counts
        from small samples are overconfident (measured: claimed 0.5 is right ~0.25 of the time)."""
        self.max_order, self.min_order = max_order, min_order
        self.min_count, self.p_min, self.cum_min, self.alpha = min_count, p_min, cum_min, alpha
        self.table: dict[tuple, dict[int, int]] = {}
        for seq in corpus:
            self.add(seq)
        self.prune()

    def add(self, seq: list[int]) -> None:
        t = self.table
        for i in range(1, len(seq)):
            nxt = seq[i]
            for n in range(self.min_order, self.max_order + 1):
                if i < n:
                    break
                d = t.setdefault(tuple(seq[i - n:i]), {})
                d[nxt] = d.get(nxt, 0) + 1

    def prune(self) -> None:
        """Drop contexts that can never fire (seen fewer than min_count times)."""
        self.table = {k: v for k, v in self.table.items() if sum(v.values()) >= self.min_count}
        # keep only the best next token: drafting is argmax-only
        self.table = {k: max(v.items(), key=lambda kv: kv[1]) + (sum(v.values()),) for k, v in self.table.items()}

    def predict(self, ctx: list[int]) -> tuple[int, float] | None:
        for n in range(min(self.max_order, len(ctx)), self.min_order - 1, -1):
            hit = self.table.get(tuple(ctx[-n:]))
            if hit:
                tok, cnt, total = hit
                return tok, cnt / (total + self.alpha)
        return None

    def propose(self, tokens: list[int], max_k: int) -> list[int]:
        if max_k <= 0 or not tokens:
            return []
        ctx = list(tokens[-self.max_order:])
        draft: list[int] = []
        cum = 1.0
        while len(draft) < max_k:
            hit = self.predict(ctx)
            if not hit or hit[1] < self.p_min or cum * hit[1] < self.cum_min:
                break
            cum *= hit[1]
            draft.append(hit[0])
            ctx = ctx[1:] + [hit[0]] if len(ctx) >= self.max_order else ctx + [hit[0]]
        return draft

    def observe(self, accepted: list[int]) -> None:
        pass  # frozen model; context-adaptive drafting is SuffixCopyProposer's job
