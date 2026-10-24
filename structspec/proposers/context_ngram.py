"""Cache language model over the current request: counts continuations seen in the
prompt and output so far (orders min_order..max_order) and drafts only where earlier
occurrences agree. Completes identifiers and repeated short idioms that a long-match
copy misses."""
from __future__ import annotations


class ContextNgramProposer:
    name = "context_ngram"

    def __init__(self, max_order: int = 4, min_order: int = 1, min_count: int = 2, p_min: float = 0.8,
                 cum_min: float = 0.3, alpha: float = 0.0):
        self.max_order, self.min_order, self.min_count = max_order, min_order, min_count
        self.p_min, self.cum_min, self.alpha = p_min, cum_min, alpha
        self._toks: list[int] = []
        self._counts: dict[tuple, dict[int, int]] = {}

    def _sync(self, tokens: list[int]) -> None:
        n0 = len(self._toks)
        if n0 > len(tokens) or (n0 and tokens[n0 - 1] != self._toks[-1]):
            self._toks, self._counts, n0 = [], {}, 0
        c = self._counts
        for i in range(max(n0, 1), len(tokens)):
            nxt = tokens[i]
            for n in range(self.min_order, self.max_order + 1):
                if i < n:
                    break
                d = c.setdefault(tuple(tokens[i - n:i]), {})
                d[nxt] = d.get(nxt, 0) + 1
        self._toks = list(tokens)

    def predict(self, ctx: list[int]) -> tuple[int, float] | None:
        for n in range(min(self.max_order, len(ctx)), self.min_order - 1, -1):
            d = self._counts.get(tuple(ctx[-n:]))
            if d:
                total = sum(d.values())
                if total < self.min_count:
                    continue  # back off: too few observations at this order
                tok, cnt = max(d.items(), key=lambda kv: kv[1])
                return tok, cnt / (total + self.alpha)
        return None

    def propose(self, tokens: list[int], max_k: int) -> list[int]:
        if max_k <= 0 or not tokens:
            return []
        self._sync(tokens)
        ctx = list(tokens[-self.max_order:])
        draft: list[int] = []
        cum = 1.0
        while len(draft) < max_k:
            hit = self.predict(ctx)
            if not hit or hit[1] < self.p_min or cum * hit[1] < self.cum_min:
                break
            cum *= hit[1]
            draft.append(hit[0])
            ctx = (ctx + [hit[0]])[-self.max_order:]
        return draft

    def observe(self, accepted: list[int]) -> None:
        pass  # counts are rebuilt incrementally from the tokens passed to propose
