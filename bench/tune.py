"""Grid-search proposer parameters on the TUNE split only (cross-fold n-gram models).

    python -m bench.tune ngram|copy|combo
Prints one line per config sorted by predicted speedup; results saved to bench/results/tune_<what>.json.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

from bench import candidates as C
from bench.replay import replay, summarize
from bench.traces import load_traces
from bench.verify_cost import load_cost
from structspec.proposers import ComboProposer, ContextNgramProposer, SuffixCopyProposer

GRIDS = {
    "ngram": dict(p_min=[0.6, 0.75, 0.9], alpha=[0, 2, 5], min_order=[2, 3, 4], cum_min=[0.2, 0.4], max_k=[8]),
    "copy": dict(min_match=[2, 3, 4, 6], len_slope=[0.5, 1, 2], len_base=[1, 2, 4], max_k=[8, 16]),
    "ctx": dict(p_min=[0.6, 0.8, 0.95], min_count=[1, 2, 3], min_order=[1, 2], cum_min=[0.3], max_k=[16]),
    "combo": dict(copy_min=[2, 4, 6, 8, 12], use_corpus=[True, False], max_k=[12, 16]),
}


def make(what: str, cfg: dict, trace: dict):
    if what == "copy":
        return SuffixCopyProposer(min_match=cfg["min_match"], len_slope=cfg["len_slope"], len_base=cfg["len_base"])
    if what == "ctx":
        return ContextNgramProposer(**{**C.PARAMS["ctx"], **{k: cfg[k] for k in cfg if k != "max_k"}})
    m = C._model_for(trace)
    for k in ("p_min", "alpha", "min_order", "cum_min"):
        setattr(m, k, cfg.get(k, C.PARAMS["ngram"].get(k, getattr(m, k))))
    if what == "ngram":
        return m
    fallbacks = [ContextNgramProposer(**C.PARAMS["ctx"])] + ([m] if cfg["use_corpus"] else [])
    return ComboProposer(SuffixCopyProposer(**C.PARAMS["copy"]), fallbacks, cfg["copy_min"])


def main() -> None:
    what = sys.argv[1]
    traces, cost = load_traces("tune"), load_cost()
    grid = GRIDS[what]
    rows = []
    for vals in itertools.product(*grid.values()):
        cfg = dict(zip(grid, vals))
        rs = [replay(make(what, cfg, t), t, cfg["max_k"], cost) for t in traces]
        s = summarize(rs)
        fresh = summarize([r for r, t in zip(rs, traces) if t["kind"] == "fresh"])
        rows.append({"cfg": cfg, "all": s, "fresh": fresh})
        print(f"{cfg} -> {s['pred_speedup']:.3f}x acc {s['acceptance']:.1%} cov {s['coverage']:.1%} "
              f"tok/pass {s['tokens_per_pass']:.2f} | fresh {fresh['pred_speedup']:.3f}x", flush=True)
    rows.sort(key=lambda r: -r["all"]["pred_speedup"])
    out = Path(__file__).parent / "results" / f"tune_{what}.json"
    out.write_text(json.dumps(rows, indent=1))
    print("BEST", rows[0]["cfg"], f"{rows[0]['all']['pred_speedup']:.3f}x")


if __name__ == "__main__":
    main()
