"""
compare_runs.py — OLD vs NEW statistical comparison report.
============================================================
Runs a fresh benchmark with current cli.py, reads an old trace CSV,
and produces a detailed side-by-side comparison.

Usage:
    python compare_runs.py --model <path.gguf> --token-json <corpus.json>
                          --old-trace <old.csv> [--prompts N] [--tokens N]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_trace(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def stats(values: list[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    return {
        "count": len(s),
        "total": sum(s),
        "mean": sum(s) / len(s),
        "median": s[len(s) // 2],
        "p90": s[int(len(s) * 0.9)] if len(s) > 1 else s[0],
        "p99": s[int(len(s) * 0.99)] if len(s) > 1 else s[0],
        "min": s[0],
        "max": s[-1],
    }


def analyze(trace: list[dict]) -> dict[str, Any]:
    rows = []
    for r in trace:
        try:
            rows.append({
                "pos": int(r.get("pos", 0)),
                "pending": int(r.get("pending", 0)),
                "draft": int(r.get("draft", 0)),
                "accepted_draft": int(r.get("accepted_draft", 0)),
                "accepted_batch": int(r.get("accepted_batch", 0)),
                "bonus": int(r.get("bonus", 0)),
                "tier": r.get("tier", "none"),
                "why": r.get("why", ""),
                "pattern_s": float(r.get("pattern_s", 0)),
                "decode_s": float(r.get("decode_s", 0)),
                "verify_s": float(r.get("verify_s", 0)),
                "rule_chain": r.get("rule_chain", ""),
                "rule_conf": r.get("rule_conf", ""),
                "rule_strikes": int(r.get("rule_strikes", 0)),
                "draft_mistakes": int(r.get("draft_mistakes", 0)),
            })
        except (ValueError, KeyError):
            continue

    if not rows:
        return {"error": "no valid rows"}

    # Phase timing
    pattern_times = [r["pattern_s"] for r in rows]
    decode_times = [r["decode_s"] for r in rows]
    verify_times = [r["verify_s"] for r in rows]

    # Draft stats
    draft_rows = [r for r in rows if r["draft"] > 0]
    proposed_total = sum(r["draft"] for r in rows)
    accepted_total = sum(r["accepted_draft"] for r in rows)
    rejected_total = proposed_total - accepted_total
    draft_starts = len(draft_rows)

    # Why categories
    why_counts = Counter(r["why"] for r in rows)

    # Tier acceptance
    tier_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in draft_rows:
        tier = r["tier"]
        tier_stats[tier][0] += r["draft"]
        tier_stats[tier][1] += r["accepted_draft"]

    tier_table = []
    for tier, (total, acc) in sorted(tier_stats.items(), key=lambda x: -x[1][0]):
        tier_table.append({
            "tier": tier,
            "proposed": total,
            "accepted": acc,
            "rate": acc / total if total else 0,
        })

    # Accuracy on no-pattern steps
    no_pattern = sum(1 for r in rows if r["draft"] == 0)
    total_steps = len(rows)

    return {
        "total_steps": total_steps,
        "total_draft_tokens_proposed": proposed_total,
        "total_draft_tokens_accepted": accepted_total,
        "total_draft_tokens_rejected": rejected_total,
        "accept_rate": accepted_total / max(1, proposed_total),
        "draft_starts": draft_starts,
        "fire_rate": draft_starts / max(1, total_steps),
        "no_pattern_steps": no_pattern,
        "pattern_timing": stats(pattern_times),
        "decode_timing": stats(decode_times),
        "verify_timing": stats(verify_times),
        "why_counts": dict(why_counts),
        "tier_table": tier_table,
    }


def run_benchmark(model: str, token_json: str, tokens: int, prompts: int, k: int, reject_mode: str, output_csv: str) -> tuple[float, str]:
    cmd = [
        sys.executable, "cli.py",
        "--model", model,
        "--token-json", token_json,
        "--tokens", str(tokens),
        "--prompts", str(prompts),
        "--k", str(k),
        "--reject-mode", reject_mode,
        "--trace-csv", output_csv,
        "--json-output",
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - t0

    # Extract JSON summary line
    summary_lines = [line for line in result.stdout.splitlines() if line.startswith("STRUCTSPEC_JSON:")]
    summary = json.loads(summary_lines[0][len("STRUCTSPEC_JSON:"):]) if summary_lines else {}
    print(result.stderr[:500] if result.stderr else "")
    return elapsed, summary


HEADER = """
================================================================================
  OLD vs NEW — SPECULATIVE DECODING COMPARISON REPORT
================================================================================
""".strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="OLD vs NEW comparison report")
    ap.add_argument("--model", required=True)
    ap.add_argument("--token-json", required=True)
    ap.add_argument("--old-trace", required=True, help="Path to old trace CSV")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prompts", type=int, default=5)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--reject-mode", default="seq-bonus")
    ap.add_argument("--output", default="comparison_report.csv")
    ap.add_argument("--no-bench", action="store_true", help="Skip benchmark, just compare traces")
    ap.add_argument("--new-trace", default="", help="Path to new trace CSV (skip running benchmark)")
    args = ap.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        raise SystemExit(2)
    if not Path(args.token_json).exists():
        print(f"ERROR: token corpus not found: {args.token_json}", file=sys.stderr)
        raise SystemExit(2)
    if not Path(args.old_trace).exists():
        print(f"ERROR: old trace not found: {args.old_trace}", file=sys.stderr)
        raise SystemExit(2)

    print(HEADER)
    print(f"\nConfig: tokens={args.tokens} prompts={args.prompts} K={args.k} reject={args.reject_mode}")

    # Load old trace
    print(f"\nLoading OLD trace: {args.old_trace}")
    old_trace = read_trace(args.old_trace)
    old_stats = analyze(old_trace)
    print(f"  Rows: {len(old_trace)}")

    # Run new benchmark
    new_csv = Path(args.output).with_suffix(".csv")
    print(f"\nRunning NEW benchmark -> {new_csv}")
    bench_time, new_summary = run_benchmark(
        args.model, args.token_json, args.tokens, args.prompts,
        args.k, args.reject_mode, str(new_csv),
    )
    print(f"  Done in {bench_time:.1f}s")

    # Load new trace
    new_trace = read_trace(new_csv)
    new_stats = analyze(new_trace)
    print(f"  Rows: {len(new_trace)}")

    # ── Comparison table ──
    print(f"\n{'='*80}")
    print(f"  {'METRIC':<40} {'OLD':>12} {'NEW':>12} {'DELTA':>12}")
    print(f"  {'-'*76}")

    metrics = [
        ("Total steps", "total_steps", "d"),
        ("Draft tokens proposed", "total_draft_tokens_proposed", "d"),
        ("Draft tokens accepted", "total_draft_tokens_accepted", "d"),
        ("Draft tokens rejected", "total_draft_tokens_rejected", "d"),
        ("Accept rate (%)", "accept_rate", ".1%"),
        ("Draft starts (fire count)", "draft_starts", "d"),
        ("Fire rate (%)", "fire_rate", ".1%"),
        ("No-pattern steps", "no_pattern_steps", "d"),
        ("Pattern time median", "pattern_timing.median", ".3f"),
        ("Pattern time total (s)", "pattern_timing.total", ".4f"),
        ("Decode time total (s)", "decode_timing.total", ".3f"),
        ("Decode time per call (ms)", "decode_timing.mean", ".2f"),
        ("Verify time total (s)", "verify_timing.total", ".4f"),
    ]

    rows_out = []
    for label, key, fmt in metrics:
        keys = key.split(".")
        old_val = old_stats
        new_val = new_stats
        for k in keys:
            old_val = old_val.get(k, 0) if isinstance(old_val, dict) else 0
            new_val = new_val.get(k, 0) if isinstance(new_val, dict) else 0
        try:
            old_val = float(old_val) if old_val else 0
            new_val = float(new_val) if new_val else 0
        except (TypeError, ValueError):
            old_val = 0
            new_val = 0

        delta = new_val - old_val
        delta_str = f"{delta:+{fmt}}" if old_val != 0 else "N/A"
        old_str = f"{old_val:{fmt}}" if old_val else "0"
        new_str = f"{new_val:{fmt}}" if new_val else "0"

        print(f"  {label:<40} {old_str:>12} {new_str:>12} {delta_str:>12}")
        rows_out.append({"metric": label, "old": old_str.strip(), "new": new_str.strip(), "delta": delta_str.strip()})

    # ── Tier comparison ──
    print(f"\n  {'TIER ACCEPTANCE COMPARISON':—^76}")
    print(f"  {'Tier':<30} {'OLD Acc/Total':>15} {'OLD %':>8} {'NEW Acc/Total':>15} {'NEW %':>8}")
    print(f"  {'-'*76}")

    old_tiers = {t["tier"]: t for t in old_stats.get("tier_table", [])}
    new_tiers = {t["tier"]: t for t in new_stats.get("tier_table", [])}
    all_tiers = sorted(set(list(old_tiers.keys()) + list(new_tiers.keys())))

    for tier in all_tiers:
        ot = old_tiers.get(tier, {"proposed": 0, "accepted": 0, "rate": 0})
        nt = new_tiers.get(tier, {"proposed": 0, "accepted": 0, "rate": 0})
        print(f"  {tier:<30} {ot['accepted']:>3}/{ot['proposed']:<3}   {ot['rate']*100:>6.1f}%   {nt['accepted']:>3}/{nt['proposed']:<3}   {nt['rate']*100:>6.1f}%")

    # ── Timing breakdown ──
    print(f"\n  {'TIMING BREAKDOWN':—^76}")
    print(f"  {'Phase':<25} {'OLD':>15} {'NEW':>15} {'Speedup':>15}")
    print(f"  {'-'*76}")

    old_pt = old_stats.get("pattern_timing", {})
    new_pt = new_stats.get("pattern_timing", {})
    old_dt = old_stats.get("decode_timing", {})
    new_dt = new_stats.get("decode_timing", {})
    old_vt = old_stats.get("verify_timing", {})
    new_vt = new_stats.get("verify_timing", {})

    for label, o, n in [
        ("Pattern mining (total)", old_pt.get("total", 0), new_pt.get("total", 0)),
        ("Pattern mining (per call)", old_pt.get("mean", 0) * 1000, new_pt.get("mean", 0) * 1000),
        ("Model decode (total)", old_dt.get("total", 0), new_dt.get("total", 0)),
        ("Model decode (per call)", old_dt.get("mean", 0), new_dt.get("mean", 0)),
        ("Verification (total)", old_vt.get("total", 0), new_vt.get("total", 0)),
    ]:
        ratio = f"{o/max(1e-9, n):.2f}x" if n > 0 else "N/A"
        print(f"  {label:<25} {o:>15.4f} {n:>15.4f} {ratio:>15}")

    # ── Rejection reason comparison ──
    print(f"\n  {'REJECTION REASONS':—^76}")
    old_why = old_stats.get("why_counts", {})
    new_why = new_stats.get("why_counts", {})
    all_why = sorted(set(list(old_why.keys()) + list(new_why.keys())))
    for why in all_why:
        print(f"  {why:<25} OLD={old_why.get(why, 0):>5}  NEW={new_why.get(why, 0):>5}")

    # ── Summary from JSON ──
    if new_summary:
        print(f"\n  {'NEW BENCHMARK JSON SUMMARY':—^76}")
        for k, v in new_summary.items():
            print(f"  {k:<30} = {v}")

    # ── Save CSV ──
    csv_out = Path(args.output)
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "old", "new", "delta"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\n  Report CSV saved to: {csv_out}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
