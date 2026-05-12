"""
diagnose.py — Fine-grained speculative decoding profiler.
============================================================
Runs ONE prompt through the speculative decoder and prints a detailed
per-step timing breakdown: pattern mining, tokenization, model decode,
and verification — every step, every token, every rejection.

Usage:
    python diagnose.py --model <path.gguf> --token-json <corpus.json>
                       [--prompt "..."] [--tokens N] [--k K]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Add cli.py's functions to namespace without importing the module-level
# llama_cpp dependency crash. We lazy-import after arg parsing.

BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
DIM = "\033[2m"


def format_time(s: float) -> str:
    if s >= 1.0:
        return f"{s:.3f}s"
    if s >= 0.001:
        return f"{s*1000:.2f}ms"
    if s >= 0.000_001:
        return f"{s*1_000_000:.1f}\u00b5s"
    return f"{s*1_000_000_000:.0f}ns"


def progress_bar(pct: float, width: int = 30) -> str:
    filled = int(width * pct)
    return "[" + "=" * filled + "-" * (width - filled) + "]"


class StepDiagnostic:
    def __init__(self, step: int, pos: int):
        self.step = step
        self.pos = pos
        self.pattern_dt = 0.0
        self.tokenize_dt = 0.0
        self.decode_dt = 0.0
        self.verify_dt = 0.0
        self.pending = 0
        self.draft_count = 0
        self.accepted = 0
        self.rejected_count = 0
        self.bonus_id = 0
        self.draft_rules: list[str] = []
        self.why = ""


def run_diagnostic(args: argparse.Namespace) -> None:
    from cli import (
        FastGreedyLlama,
        PatternMiner,
        PythonSyntaxProposer,
        QwenTokenCorpus,
        propose_draft,
    )

    model = FastGreedyLlama(
        str(args.model),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        flash_attn=args.flash_attn,
    )

    corpus = QwenTokenCorpus(args.token_json)
    miner = PatternMiner(
        corpus.token_text,
        max_ctx=args.max_ctx,
        min_support=args.min_support,
        min_conf=args.min_conf,
        det_conf=args.det_conf,
        min_rule_ctx=args.min_rule_ctx,
    ).fit(corpus.sequences)

    syntax = (
        None
        if args.no_syntax
        else PythonSyntaxProposer(model, mode=args.syntax_mode)
    )

    if syntax is not None:
        syntax.reset_cache()
        syntax.feed_token(args.prompt)

    model.reset()
    prompt_ids = model.tokenize(args.prompt)

    t0 = time.perf_counter()
    logits = model.decode_logits(prompt_ids, logits_all=False)[0]
    prompt_dt = time.perf_counter() - t0

    gen = list(prompt_ids)
    kv_len = len(prompt_ids)
    target = len(prompt_ids) + args.tokens
    passes = 1
    prev_pred = model.argmax(logits)

    steps: list[StepDiagnostic] = []
    total = Counter()
    banned_rules: set = set()
    rule_strikes: Counter = Counter()

    print(f"\n{BOLD}{'═'*78}{RESET}")
    print(f"{BOLD}  STRUCTSPEC DIAGNOSTIC PROFILER{RESET}")
    print(f"{BOLD}{'═'*78}{RESET}")
    print(f"  Model    : {args.model}")
    print(f"  Prompt   : {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"  Tokens   : {args.tokens}    K = {args.k}    Reject mode = {args.reject_mode}")
    print(f"  Syntax   : {'ON' if syntax else 'OFF'} (mode={args.syntax_mode})")
    print(f"  Prompt   : {format_time(prompt_dt)} for {len(prompt_ids)} tokens")
    print(f"{'─'*78}")

    step_idx = 0
    start_wall = time.perf_counter()

    while len(gen) < target:
        if gen[-1] == model.eos:
            break

        remaining = target - len(gen)
        if remaining <= 0:
            break

        step_idx += 1
        diag = StepDiagnostic(step_idx, len(gen) - len(prompt_ids))
        pending = gen[kv_len:]

        if len(pending) > 1:
            print(f"\n  {RED}[WARN] pending > 1 at pos {len(gen)}{RESET}")
            gen = gen[:kv_len]
            pending = []
            model.truncate_kv(kv_len)
            if gen:
                recover_logits = model.decode_logits(gen[-1:], logits_all=False)[0]
                prev_pred = model.argmax(recover_logits)
                passes += 1

        max_draft = max(0, min(args.k, remaining - 1))

        # ── Pattern mining timing ──────────────────────────────────────
        t_pat = time.perf_counter()
        draft_ids, rules = propose_draft(
            miner, syntax, gen, max_draft,
            banned=banned_rules,
            adaptive_k=False,
        )
        diag.draft_rules = [r.tier for r in rules]
        diag.draft_count = len(draft_ids)

        # ── Tokenization timing (feed syntax/ast) ───────────────────────
        t_tok = time.perf_counter()
        if not pending and not draft_ids:
            gen.append(prev_pred)
            if syntax:
                syntax.feed_token(model.token_piece(prev_pred))
            if prev_pred == model.eos or len(gen) >= target:
                break
            continue

        batch = pending + draft_ids
        if not batch:
            gen.append(prev_pred)
            if syntax:
                syntax.feed_token(model.token_piece(prev_pred))
            continue
        diag.tokenize_dt = time.perf_counter() - t_tok
        diag.pending = len(pending)

        old_kv = kv_len

        # ── Model decode timing ────────────────────────────────────────
        t_dec = time.perf_counter()
        import numpy as np

        batch_logits = model.decode_logits(batch, logits_all=len(batch) > 1)
        diag.decode_dt = time.perf_counter() - t_dec
        passes += 1
        total["decode_calls"] += 1
        total["decode_tokens"] += len(batch)

        # ── Verification timing ────────────────────────────────────────
        t_ver = time.perf_counter()
        preds = np.empty(len(batch) + 1, dtype=np.intc)
        preds[0] = prev_pred
        if len(batch) > 1:
            preds[1:-1] = np.argmax(batch_logits[:-1], axis=1).astype(np.intc, copy=False)
        preds[-1] = model.argmax(batch_logits[-1])

        accepted_batch = 0
        for i, tok in enumerate(batch):
            if int(preds[i]) == tok:
                accepted_batch += 1
            else:
                break
        bonus = int(preds[accepted_batch])
        model_draft = [int(preds[len(pending) + j]) for j in range(len(draft_ids)) if len(pending) + j < len(preds)]

        rejected = accepted_batch < len(batch)
        accept_draft = max(0, accepted_batch - len(pending))

        why = "draft_accepted" if draft_ids else "no_pattern"
        mismatch_at: int | str = ""
        if draft_ids:
            for j, (want, got) in enumerate(zip(draft_ids, model_draft)):
                if want != got:
                    mismatch_at = j
                    why = "draft_mismatch"
                    key = (rules[j].ctx if j < len(rules) else (), rules[j].token if j < len(rules) else 0, rules[j].tier if j < len(rules) else "")
                    rule_strikes[key] += 1
                    if rule_strikes[key] >= 3:
                        banned_rules.add(key)
                    break
        if rejected and accepted_batch < len(pending):
            why = "bonus_mismatch"

        if rejected and args.reject_mode == "seq-bonus":
            if accepted_batch > 0:
                last_tok = batch[accepted_batch - 1]
                model.truncate_kv(old_kv + accepted_batch - 1)
                seq_logits = model.decode_logits([last_tok], logits_all=False)[0]
                total["decode_calls"] += 1
                total["decode_tokens"] += 1
                passes += 1
                bonus = model.argmax(seq_logits)
            else:
                model.truncate_kv(old_kv)
                bonus = prev_pred
        elif rejected and args.reject_mode == "truncate":
            pass

        diag.verify_dt = time.perf_counter() - t_ver
        diag.accepted = accept_draft
        diag.rejected_count = len(draft_ids) - accept_draft
        diag.bonus_id = bonus
        diag.why = why

        total["proposed"] += len(draft_ids)
        total["accepted"] += accept_draft
        total["rejected"] += (len(draft_ids) - accept_draft)
        total["draft_starts"] += int(bool(draft_ids))
        total["reject_events"] += int(rejected)
        total[f"why_{why}"] += 1

        gen = gen[:old_kv] + batch[:accepted_batch] + [bonus]
        kv_len = old_kv + accepted_batch
        model.truncate_kv(kv_len)
        prev_pred = bonus

        if syntax:
            for tok in batch[:accepted_batch]:
                syntax.feed_token(model.token_piece(tok))
            syntax.feed_token(model.token_piece(bonus))

        steps.append(diag)

    wall_dt = time.perf_counter() - start_wall

    # ── Detailed step-by-step report ───────────────────────────────────
    print(f"\n{BOLD}PER-STEP TIMING BREAKDOWN{RESET}")
    print(f"{'Step':>4} {'Pos':>4} {'Pend':>4} {'Draft':>5} {'Acc':>3} {'Rej':>3} "
          f"{'Pattern':>10} {'Decode':>10} {'Verify':>10} {'Total':>10} {'Rules'}")
    print("─" * 100)

    for d in steps:
        total_dt = d.pattern_dt + d.tokenize_dt + d.decode_dt + d.verify_dt
        rules_short = ",".join(r.replace("syntax_", "s:")[:12] for r in d.draft_rules[:3])
        if len(d.draft_rules) > 3:
            rules_short += f" +{len(d.draft_rules)-3}"
        print(
            f"{d.step:>4} {d.pos:>4} {d.pending:>4} {d.draft_count:>5} {d.accepted:>3} {d.rejected_count:>3} "
            f"{format_time(d.pattern_dt):>10} {format_time(d.decode_dt):>10} "
            f"{format_time(d.verify_dt):>10} {format_time(total_dt):>10} "
            f"{rules_short}"
        )

    # ── Time allocation summary ────────────────────────────────────────
    print(f"\n{BOLD}TIME ALLOCATION SUMMARY{RESET}")
    pattern_total = sum(d.pattern_dt for d in steps)
    decode_total = sum(d.decode_dt for d in steps)
    verify_total = sum(d.verify_dt for d in steps)
    tokenize_total = sum(d.tokenize_dt for d in steps)
    grand_total = pattern_total + decode_total + verify_total + tokenize_total

    print(f"  {'Phase':<20} {'Time':>12} {'%':>8}  {'Distribution'}")
    print("  " + "─" * 70)
    for label, val in [
        ("Pattern mining", pattern_total),
        ("Model decode", decode_total),
        ("Verification", verify_total),
        ("Tokenization", tokenize_total),
        ("TOTAL spec time", grand_total),
    ]:
        pct = val / max(1e-9, grand_total)
        bar = progress_bar(pct, width=30)
        print(f"  {label:<20} {format_time(val):>12} {pct*100:>7.1f}%  {CYAN}{bar}{RESET}")

    # ── Decode cost per token ──────────────────────────────────────────
    print(f"\n{BOLD}MODEL DECODE COST PER TOKEN{RESET}")
    decode_tokens = total["decode_tokens"]
    decode_calls = total["decode_calls"]
    if decode_calls > 0:
        avg_tokens_per_call = decode_tokens / decode_calls
        avg_time_per_call = decode_total / decode_calls
        avg_time_per_token = decode_total / max(1, decode_tokens)
        print(f"  Total decode calls    : {decode_calls}")
        print(f"  Total tokens decoded  : {decode_tokens}")
        print(f"  Avg tokens per call   : {avg_tokens_per_call:.1f}")
        print(f"  Avg time per call     : {format_time(avg_time_per_call)}")
        print(f"  Avg time per token    : {format_time(avg_time_per_token)}")
        print(f"  Total decode time     : {format_time(decode_total)}")

    # ── Draft acceptance analysis ──────────────────────────────────────
    print(f"\n{BOLD}DRAFT ACCEPTANCE ANALYSIS{RESET}")
    proposed = total["proposed"]
    accepted = total["accepted"]
    rejected = total["rejected"]
    draft_starts = total["draft_starts"]
    if proposed > 0:
        accept_rate = accepted / proposed
        reject_rate = rejected / proposed
        print(f"  Draft proposals       : {draft_starts}")
        print(f"  Draft tokens proposed : {proposed}")
        print(f"  Draft tokens accepted : {accepted}  ({GREEN}{accept_rate*100:.1f}%{RESET})")
        print(f"  Draft tokens rejected : {rejected}  ({RED}{reject_rate*100:.1f}%{RESET})")
        bar = progress_bar(accept_rate, width=30)
        color = GREEN if accept_rate > 0.5 else YELLOW if accept_rate > 0.3 else RED
        print(f"  Acceptance  {color}{bar}{RESET} {accept_rate*100:.1f}%")
    else:
        print(f"  {YELLOW}No drafts proposed — miner/syntax never fired.{RESET}")

    # ── Rejection reason breakdown ─────────────────────────────────────
    print(f"\n{BOLD}REJECTION REASONS{RESET}")
    for why in ["draft_accepted", "draft_mismatch", "no_pattern", "bonus_mismatch"]:
        count = total.get(f"why_{why}", 0)
        if count > 0:
            color = GREEN if why == "draft_accepted" else RED if "mismatch" in why else YELLOW
            print(f"  {color}{why:<20}{RESET} {count:>5}")

    # ── Per-pattern-tier acceptance ────────────────────────────────────
    tier_acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for d in steps:
        for i, tier in enumerate(d.draft_rules):
            tier_acc[tier][0] += 1
            if i < d.accepted:
                tier_acc[tier][1] += 1
    if tier_acc:
        print(f"\n{BOLD}ACCEPTANCE BY PATTERN TIER{RESET}")
        print(f"  {'Tier':<30} {'Accepted/Total':>15} {'Rate':>8}")
        print("  " + "─" * 56)
        for tier, (total_count, accepted_count) in sorted(tier_acc.items(), key=lambda x: -x[1][0]):
            rate = accepted_count / total_count if total_count > 0 else 0
            color = GREEN if rate > 0.6 else YELLOW if rate > 0.3 else RED
            print(f"  {tier:<30} {color}{accepted_count:>3}/{total_count:<3}{RESET}     {rate*100:>6.1f}%")

    # ── Speedup summary ────────────────────────────────────────────────
    print(f"\n{BOLD}SPEEDUP SUMMARY{RESET}")
    gen_tokens = len(gen) - len(prompt_ids)
    gen_text = model.detokenize(gen[len(prompt_ids):][:args.tokens])
    print(f"  Tokens generated      : {gen_tokens}")
    print(f"  Model passes          : {passes}")
    print(f"  Tokens per pass       : {gen_tokens/max(1, passes):.2f}")
    print(f"  Wall time             : {format_time(wall_dt)}")
    print(f"  Tokens/sec            : {gen_tokens/max(1e-9, wall_dt):.1f}")
    print(f"  Prompt eval           : {format_time(prompt_dt)}")
    newline = chr(10)
    escaped = "\\n"
    print(f"  Output preview        : {gen_text[:120].replace(newline, escaped)}...")
    print(f"\n{BOLD}{'═'*78}{RESET}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Structspec diagnostic profiler")
    ap.add_argument("--model", required=True, help="Path to GGUF model")
    ap.add_argument("--token-json", required=True, help="Path to token corpus JSON")
    ap.add_argument("--prompt", default="implement linked list in python only code no comments", help="Prompt text")
    ap.add_argument("--tokens", type=int, default=64, help="Max tokens to generate")
    ap.add_argument("--k", type=int, default=6, help="Max draft tokens per step")
    ap.add_argument("--max-ctx", type=int, default=8)
    ap.add_argument("--min-support", type=int, default=1)
    ap.add_argument("--min-conf", type=float, default=0.85)
    ap.add_argument("--det-conf", type=float, default=0.96)
    ap.add_argument("--min-rule-ctx", type=int, default=4)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--flash-attn", action="store_true")
    ap.add_argument("--no-syntax", action="store_true", help="Disable syntax proposer")
    ap.add_argument("--syntax-mode", default="cluster", choices=["off", "basic", "cluster"])
    ap.add_argument("--reject-mode", default="seq-bonus", choices=["truncate", "seq-bonus", "rebuild"])
    args = ap.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        raise SystemExit(2)
    if not Path(args.token_json).exists():
        print(f"ERROR: token json not found: {args.token_json}", file=sys.stderr)
        raise SystemExit(2)

    run_diagnostic(args)


if __name__ == "__main__":
    main()
