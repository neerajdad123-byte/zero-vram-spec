"""Proposer factories for replay: name -> make(trace) -> Proposer.

The learned n-gram model is trained on the mining corpus plus *tune* traces. While
tuning, the tune split is cut into two problem-level folds and each fold is scored with
a model that never saw it; on the test split the model sees all tune traces.
"""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from bench.corpus import mining_tokens, vocab_pieces
from bench.traces import load_traces
from structspec.proposers import (
    ComboProposer,
    ContextNgramProposer,
    LineStartProposer,
    NgramModelProposer,
    SuffixCopyProposer,
)

V2CORE_REF = "5c022a8"  # v2-core commit whose proposers are the in-repo baseline
# tuned on the tune split by bench/tune.py (results in bench/results/tune_*.json)
PARAMS: dict = {"ngram": dict(max_order=6, min_count=2, p_min=0.75, cum_min=0.4),
                "copy": dict(min_match=2, len_slope=0.5, len_base=1),
                "ctx": dict(max_order=4, min_order=2, min_count=1, p_min=0.8, cum_min=0.3), "copy_min": 2,
                "line": dict(min_count=2, p_min=0.6)}


def problem_of(trace_id: str) -> str:
    if trace_id.startswith("HumanEval/"):
        return "he" + trace_id.split("/")[1]
    if trace_id.startswith("multiple/"):
        return "he" + trace_id.split("/")[1].split("_")[1]
    return trace_id


def fold_of(trace_id: str) -> int:
    return int(hashlib.sha1(("fold" + problem_of(trace_id)).encode()).hexdigest(), 16) % 2


@lru_cache(maxsize=8)
def ngram(holdout_fold: int | None, use_mining: bool = True, use_traces: bool = True, **kw) -> NgramModelProposer:
    """holdout_fold: tune fold to leave out (None = use all tune traces, for the test split)."""
    seqs = [m["ids"] for m in mining_tokens()] if use_mining else []
    if use_traces:
        seqs += [t["out_ids"] for t in load_traces("tune")
                 if holdout_fold is None or fold_of(t["id"]) != holdout_fold]
    return NgramModelProposer(seqs, **(kw or PARAMS["ngram"]))


def _model_for(trace: dict, **kw) -> NgramModelProposer:
    # a trace in the tune split must be scored by the model trained without its fold
    tune_ids = _tune_ids()
    hold = fold_of(trace["id"]) if trace["id"] in tune_ids else None
    return ngram(hold, **kw)


@lru_cache(maxsize=4)
def _line_table(holdout_fold: int | None) -> LineStartProposer:
    seqs = [m["ids"] for m in mining_tokens()] + [t["out_ids"] for t in load_traces("tune")
                                                   if holdout_fold is None or fold_of(t["id"]) != holdout_fold]
    return LineStartProposer(vocab_pieces(), seqs, **PARAMS["line"])


def _line_for(trace: dict, extend=None) -> LineStartProposer:
    base = _line_table(_hold(trace))
    p = LineStartProposer(base.pieces, (), base.min_count, base.p_min, extend)
    p.table = base.table
    return p


@lru_cache(maxsize=1)
def _tune_ids() -> frozenset:
    return frozenset(t["id"] for t in load_traces("tune"))


@lru_cache(maxsize=1)
def _v2core():
    src = subprocess.run(["git", "show", f"{V2CORE_REF}:structspec/engine.py"], capture_output=True, text=True,
                         check=True, cwd=Path(__file__).parent).stdout
    spec = importlib.util.spec_from_loader("v2core_engine", loader=None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v2core_engine"] = mod  # dataclasses look the module up while exec runs
    exec(compile(src, "v2core_engine.py", "exec"), mod.__dict__)
    return mod


class V2CorePromptLookup:
    """Copy of v2-core's PromptLookupProposer (structspec/proposer.py @ 5c022a8): 3..2-gram, copy max_k."""
    name = "v2core_prompt_lookup"

    def propose(self, tokens, max_k):
        size = len(tokens)
        for n in range(min(3, size - 1), 1, -1):
            pattern = tokens[-n:]
            for start in range(size - n - 1, -1, -1):
                if tokens[start:start + n] == pattern:
                    return list(tokens[start + n:start + n + max_k])
        return []

    def observe(self, accepted):
        pass


@lru_cache(maxsize=4)
def _v2core_miner(holdout_fold):
    eng = _v2core()
    seqs = [m["ids"] for m in mining_tokens()] + [t["out_ids"] for t in load_traces("tune")
                                                   if holdout_fold is None or fold_of(t["id"]) != holdout_fold]
    return eng.PatternMiner(max_ctx=8, min_support=2, min_conf=0.9, min_rule_ctx=3).fit(seqs)


class V2CoreIndex:
    name = "v2core_ngram_index"

    def __init__(self, miner):
        self.miner = miner

    def propose(self, tokens, max_k):
        return self.miner.propose(tokens, max_k)[0]

    def observe(self, accepted):
        pass


class _First:
    """First proposer that drafts wins (all observe)."""
    name = "first"

    def __init__(self, ps):
        self.ps = ps

    def propose(self, tokens, max_k):
        for p in self.ps:
            d = p.propose(tokens, max_k)
            if d:
                return d
        return []

    def observe(self, accepted):
        for p in self.ps:
            p.observe(accepted)


def _hold(trace):
    return fold_of(trace["id"]) if trace["id"] in _tune_ids() else None


REGISTRY = {
    "v2core_prompt_lookup": lambda t: V2CorePromptLookup(),
    "v2core_ngram_index": lambda t: V2CoreIndex(_v2core_miner(_hold(t))),
    "suffix_copy": lambda t: SuffixCopyProposer(**PARAMS["copy"]),
    "ngram_model": lambda t: _model_for(t),
    "context_ngram": lambda t: ContextNgramProposer(**PARAMS["ctx"]),
    "combo": lambda t: ComboProposer(SuffixCopyProposer(**PARAMS["copy"]),
                                     [ContextNgramProposer(**PARAMS["ctx"]), _model_for(t)], PARAMS["copy_min"]),
    "line_start": lambda t: _line_for(t, _model_for(t)),
    "combo_line_after_copy": lambda t: ComboProposer(SuffixCopyProposer(**PARAMS["copy"]),
                                                     [_line_for(t, _model_for(t)), _model_for(t)], PARAMS["copy_min"]),
    "combo_line_first": lambda t: _First([_line_for(t, _model_for(t)), ComboProposer(
        SuffixCopyProposer(**PARAMS["copy"]), [_model_for(t)], PARAMS["copy_min"])]),
    "combo_noncorpus": lambda t: ComboProposer(SuffixCopyProposer(**PARAMS["copy"]),
                                               [ContextNgramProposer(**PARAMS["ctx"])], PARAMS["copy_min"]),
}
