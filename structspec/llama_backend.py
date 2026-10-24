from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np

from .proposer import Proposer

REJECT_MODES = ("truncate", "seq-bonus", "rebuild")


def now() -> float:
    return time.perf_counter()


@dataclass
class SpecStats:
    passes: int = 0
    generated: int = 0
    proposed: int = 0
    accepted: int = 0
    draft_starts: int = 0


def _prompt_ids(model, prompt: str | list[int]) -> list[int]:
    return list(prompt) if isinstance(prompt, list) else model.encode(prompt)


def speculate(
    model,
    proposer: Proposer,
    prompt_ids: list[int],
    max_tokens: int,
    k: int = 4,
    reject_mode: str = "truncate",
    stats: SpecStats | None = None,
) -> Iterator[list[int]]:
    """Greedy speculative decoding. Yields the tokens committed by each target pass.

    Output equals plain greedy decoding: a draft token is kept only if it is the target's
    argmax at that position, and generation stops after the first end-of-generation token.
    """
    if reject_mode not in REJECT_MODES:
        raise ValueError(f"reject_mode must be one of {REJECT_MODES}")
    stats = stats if stats is not None else SpecStats()
    if max_tokens <= 0:
        return
    model.reset()
    pred = model.argmax(model.decode_logits(prompt_ids, logits_all=False)[0])
    stats.passes += 1
    gen = list(prompt_ids)  # committed tokens; the KV cache holds exactly these
    budget = max_tokens

    while True:
        if model.is_eog(pred) or budget == 1:
            stats.generated += 1
            proposer.observe([pred])
            yield [pred]
            return

        max_draft = min(k, budget - 1)
        draft = [int(t) for t in proposer.propose(gen + [pred], max_draft)[:max_draft]] if max_draft > 0 else []
        batch = [pred] + draft
        rows = model.decode_logits(batch, logits_all=len(batch) > 1)
        stats.passes += 1
        preds = np.argmax(rows, axis=1)

        n = 0
        while n < len(draft) and int(preds[n]) == draft[n]:
            n += 1
        committed = batch[: n + 1]
        new_pred = int(preds[n])

        if n < len(draft):  # roll back the rejected tail
            if reject_mode == "truncate":
                model.truncate_kv(len(gen) + n + 1)
            elif reject_mode == "seq-bonus":
                # recompute the bonus token with a batch of one, as plain greedy would
                model.truncate_kv(len(gen) + n)
                new_pred = model.argmax(model.decode_logits([committed[-1]], logits_all=False)[0])
                stats.passes += 1
            else:  # rebuild
                model.reset()
                new_pred = model.argmax(model.decode_logits(gen + committed, logits_all=False)[0])
                stats.passes += 1

        eog_at = next((i for i, t in enumerate(committed) if model.is_eog(t)), None)
        if eog_at is not None:
            committed = committed[: eog_at + 1]

        stats.proposed += len(draft)
        stats.accepted += n
        stats.draft_starts += int(bool(draft))
        stats.generated += len(committed)
        gen += committed
        budget -= len(committed)
        proposer.observe(committed)
        yield committed
        if eog_at is not None or budget == 0:
            return
        pred = new_pred


def run_speculative(
    model,
    proposer: Proposer,
    prompt: str | list[int],
    max_tokens: int,
    k: int = 4,
    reject_mode: str = "truncate",
) -> tuple[list[int], str, dict]:
    prompt_ids = _prompt_ids(model, prompt)
    stats = SpecStats()
    start = now()
    out = [t for chunk in speculate(model, proposer, prompt_ids, max_tokens, k, reject_mode, stats) for t in chunk]
    elapsed = now() - start
    result = asdict(stats)
    result.update(time=elapsed, hit_rate=stats.accepted / stats.proposed if stats.proposed else 0.0)
    return prompt_ids + out, model.decode(out), result


def run_greedy(model, prompt: str | list[int], max_tokens: int) -> tuple[list[int], str, dict]:
    prompt_ids = _prompt_ids(model, prompt)
    model.reset()
    start = now()
    out: list[int] = []
    passes = 0
    if max_tokens > 0:
        logits = model.decode_logits(prompt_ids, logits_all=False)[0]
        passes = 1
        while True:
            tok = model.argmax(logits)
            out.append(tok)
            if model.is_eog(tok) or len(out) >= max_tokens:
                break
            logits = model.decode_logits([tok], logits_all=False)[0]
            passes += 1
    elapsed = now() - start
    return prompt_ids + out, model.decode(out), {"time": elapsed, "passes": passes, "generated": len(out)}
