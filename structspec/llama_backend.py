from __future__ import annotations

import sys
import time
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

try:
    from llama_cpp import Llama, LlamaCache
except ImportError as _imp_err:  # pragma: no cover
    Llama = None  # type: ignore[misc,assignment]
    LlamaCache = None  # type: ignore[misc,assignment]
    _LLAMA_CPP_MISSING = _imp_err
else:
    _LLAMA_CPP_MISSING = None

from .engine import (
    AdaptiveKController,
    EarlyExitDraftController,
    EntropyProxyAbort,
    LastTargetTopKFilter,
    PatternMiner,
    PythonSyntaxProposer,
    RuleStatsRegistry,
    SpeedTraceCollector,
    propose_draft,
)

if TYPE_CHECKING:
    pass


def now() -> float:
    return time.perf_counter()


class FastGreedyLlama:
    """Thin, fast wrapper around llama-cpp-python for greedy speculative decoding."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1):
        if Llama is None:
            raise _LLAMA_CPP_MISSING  # type: ignore[truthy-function]
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            logits_all=False,
            verbose=False,
        )
        self.llm.set_cache(LlamaCache())
        self.n_vocab = self.llm.n_vocab()
        self.eos = self.llm.token_eos()
        self._piece_cache: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def decode(self, toks: list[int]) -> str:
        if not toks:
            return ""
        return self.llm.detokenize(toks).decode("utf-8", errors="ignore")

    def piece(self, tok: int) -> str:
        tok = int(tok)
        piece = self._piece_cache.get(tok)
        if piece is None:
            piece = self.decode([tok])
            self._piece_cache[tok] = piece
        return piece

    def reset(self) -> None:
        self.llm.reset()

    def truncate_kv(self, n_tokens: int) -> None:
        self.llm.n_tokens = n_tokens
        self.llm._ctx.kv_cache_seq_rm(-1, n_tokens, -1)

    def decode_logits(self, tokens: list[int], logits_all: bool) -> np.ndarray:
        if not tokens:
            raise ValueError("decode_logits requires at least one token")
        self.llm._ctx.kv_cache_seq_rm(-1, self.llm.n_tokens, -1)
        out = None
        for i in range(0, len(tokens), self.llm.n_batch):
            batch = tokens[i : min(len(tokens), i + self.llm.n_batch)]
            n_past = self.llm.n_tokens
            self.llm._batch.set_batch(batch=batch, n_past=n_past, logits_all=logits_all)
            self.llm._ctx.decode(self.llm._batch)
            self.llm.input_ids[n_past : n_past + len(batch)] = batch
            rows = len(batch) if logits_all else 1
            ptr = self.llm._ctx.get_logits()
            out = np.ctypeslib.as_array(ptr, shape=(rows * self.n_vocab,)).reshape(
                rows, self.n_vocab
            )
            self.llm.n_tokens += len(batch)
        assert out is not None
        return out

    @staticmethod
    def argmax(row: np.ndarray) -> int:
        return int(np.argmax(row))


def blank_times() -> dict[str, float]:
    return {"prompt": 0.0, "pattern": 0.0, "decode": 0.0, "verify": 0.0, "detok": 0.0}


def run_greedy(model: FastGreedyLlama, prompt: str, max_tokens: int) -> tuple[list[int], str, dict]:
    times = blank_times()
    records = []
    model.reset()
    prompt_ids = model.encode(prompt)

    t0 = now()
    logits = model.decode_logits(prompt_ids, logits_all=False)[0]
    times["prompt"] += now() - t0
    prev_pred = model.argmax(logits)

    gen = list(prompt_ids)
    passes = 1
    target = len(prompt_ids) + max_tokens
    start = now()

    while len(gen) < target:
        tok = prev_pred
        gen.append(tok)
        if tok == model.eos or len(gen) >= target:
            break
        t0 = now()
        logits = model.decode_logits([tok], logits_all=False)[0]
        dt = now() - t0
        times["decode"] += dt
        passes += 1
        prev_pred = model.argmax(logits)
        records.append({"kind": "greedy_eval", "tokens": 1, "time": dt})

    elapsed = now() - start
    out_ids = gen[:target]
    t0 = now()
    text = model.decode(out_ids[len(prompt_ids) :])
    times["detok"] += now() - t0
    return out_ids, text, {
        "time": elapsed,
        "passes": passes,
        "target_tokens_evaluated": passes - 1,
        "times": times,
        "records": records,
    }


def run_speculative(
    model: FastGreedyLlama,
    miner: PatternMiner,
    syntax: PythonSyntaxProposer | None,
    prompt: str,
    max_tokens: int,
    k: int = 6,
    reject_mode: str = "truncate",
    strike_limit: int = 3,
    draft_model: FastGreedyLlama | None = None,
    draft_mistakes: Counter[tuple[tuple[int, ...], int]] | None = None,
    draft_mistake_limit: int = 1,
    live_viz: bool = False,
    rule_registry: RuleStatsRegistry | None = None,
    adaptive_k: AdaptiveKController | None = None,
    top_k_filter: LastTargetTopKFilter | None = None,
    early_exit: EarlyExitDraftController | None = None,
    entropy_abort: EntropyProxyAbort | None = None,
    speed_trace: SpeedTraceCollector | None = None,
) -> tuple[list[int], str, dict]:
    times = blank_times()
    rows = []
    model.reset()
    prompt_ids = model.encode(prompt)

    t0 = now()
    logits = model.decode_logits(prompt_ids, logits_all=False)[0]
    times["prompt"] += now() - t0
    prev_pred = model.argmax(logits)

    gen = list(prompt_ids)
    kv_len = len(prompt_ids)
    target = len(prompt_ids) + max_tokens
    passes = 1

    proposed = accepted = draft_starts = greedy_emitted = 0
    reject_count = 0
    reject_time_total = 0.0
    pattern_time_total = 0.0
    verify_time_total = 0.0
    accept_hist = Counter()
    banned_rules: dict[tuple[tuple[int, ...], int, str], int] = {}
    rule_strikes: Counter[tuple[tuple[int, ...], int, str]] = Counter()
    draft_mistakes = draft_mistakes if draft_mistakes is not None else Counter()
    BAN_DECAY = 50

    rule_registry = rule_registry or RuleStatsRegistry()
    adaptive_k = adaptive_k or AdaptiveKController(k_init=k)
    top_k_filter = top_k_filter or LastTargetTopKFilter(top_k=5)
    early_exit = early_exit or EarlyExitDraftController()
    entropy_abort = entropy_abort or EntropyProxyAbort()
    live_ngram = None  # DISABLED: causes more rejections than accepted drafts

    start = now()

    while len(gen) < target:
        if gen and gen[-1] == model.eos:
            break

        remaining = target - len(gen)
        if remaining <= 0:
            break

        pending = gen[kv_len:]
        if len(pending) > 1:
            print(
                f"\n[warn] pending={pending} > 1 at position {len(gen)}, "
                f"falling back to greedy for this step",
                file=sys.stderr,
                flush=True,
            )
            gen = gen[:kv_len]
            pending = []
            model.truncate_kv(kv_len)
            if gen:
                recover_logits = model.decode_logits(gen[-1:], logits_all=False)[0]
                times["decode"] += now() - (now() - 0)
                passes += 1
                prev_pred = model.argmax(recover_logits)

        k_now = adaptive_k.get_k() if adaptive_k is not None else k
        max_draft = max(0, min(k_now, remaining - 1))
        ctx_tail_ids = gen[max(0, len(gen) - 12) :]
        pos = len(gen) - len(prompt_ids)

        expired = [key for key, ban_pos in banned_rules.items() if pos - ban_pos > BAN_DECAY]
        for key in expired:
            del banned_rules[key]

        t0 = now()
        draft, rules = propose_draft(
            miner,
            syntax,
            gen,
            max_draft,
            banned=banned_rules,
            early_exit=early_exit,
            rule_registry=rule_registry,
            live_ngram=live_ngram,
        )
        if not draft and draft_model is not None and max_draft > 0:
            draft, rules = _propose_from_draft_model(
                draft_model, gen, max_draft, draft_mistakes, draft_mistake_limit
            )
        pattern_dt = now() - t0
        times["pattern"] += pattern_dt
        pattern_time_total += pattern_dt

        if draft and rules:
            first_rule = rules[0]
            rkey = f"{first_rule.tier}:{first_rule.token}"
            rstats = rule_registry.get_or_create(rkey, first_rule.confidence)
            if not rstats.should_fire():
                draft, rules = [], []

        if not pending and not draft:
            gen.append(prev_pred)
            greedy_emitted += 1
            if prev_pred == model.eos or len(gen) >= target:
                break
            continue

        batch = pending + draft
        if not batch:
            gen.append(prev_pred)
            greedy_emitted += 1
            continue

        old_kv = kv_len
        t0 = now()
        batch_logits = model.decode_logits(batch, logits_all=len(batch) > 1)
        decode_dt = now() - t0
        times["decode"] += decode_dt
        passes += 1

        t0 = now()
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
        model_draft = [
            int(preds[len(pending) + j])
            for j in range(len(draft))
            if len(pending) + j < len(preds)
        ]
        verify_dt = now() - t0
        times["verify"] += verify_dt
        verify_time_total += verify_dt

        rejected = accepted_batch < len(batch)
        accepted_pending = min(accepted_batch, len(pending))
        accepted_draft = max(0, accepted_batch - len(pending))

        mismatch_at: int | str = ""
        why = "draft_accepted" if draft else "no_pattern_fired"
        if draft:
            for j, (want, got) in enumerate(zip(draft, model_draft)):
                if want != got:
                    mismatch_at = j
                    why = "draft_token_mismatch"
                    if j < len(rules):
                        key = PatternMiner.rule_key(rules[j])
                        rule_strikes[key] += 1
                        if rule_strikes[key] >= strike_limit:
                            banned_rules[key] = pos
                        if rules[j].tier == "draft_model":
                            draft_mistakes[(rules[j].ctx, rules[j].token)] += 1
                    break
        if rejected and accepted_batch < len(pending):
            why = "pending_bonus_mismatch"

        rec_time = 0.0
        if rejected and reject_mode == "rebuild":
            accepted_batch = accepted_pending
            accepted_draft = 0
            clean_prefix = gen[:old_kv] + batch[:accepted_batch]
            model.reset()
            t1 = now()
            clean_logits = model.decode_logits(prompt_ids, logits_all=False)[0]
            rebuild_passes = 1
            for replay_tok in clean_prefix[len(prompt_ids) :]:
                clean_logits = model.decode_logits([replay_tok], logits_all=False)[0]
                rebuild_passes += 1
            rebuild_dt = now() - t1
            times["decode"] += rebuild_dt
            decode_dt += rebuild_dt
            passes += rebuild_passes
            bonus = model.argmax(clean_logits)
            rec_time = rebuild_dt
        elif rejected and reject_mode == "seq-bonus":
            if accepted_batch > 0:
                last_accepted_tok = batch[accepted_batch - 1]
                model.truncate_kv(old_kv + accepted_batch - 1)
                t1 = now()
                seq_logits = model.decode_logits([last_accepted_tok], logits_all=False)[0]
                seq_dt = now() - t1
                times["decode"] += seq_dt
                decode_dt += seq_dt
                passes += 1
                bonus = model.argmax(seq_logits)
                rec_time = seq_dt
            else:
                model.truncate_kv(old_kv)
                bonus = prev_pred
                rec_time = 0.0
        elif rejected and reject_mode == "truncate":
            rec_time = 0.0

        reject_time_total += rec_time

        if top_k_filter is not None:
            top_k_filter.update(batch_logits[-1])

        if draft and rules:
            first_rule = rules[0]
            rkey = f"{first_rule.tier}:{first_rule.token}"
            rstats = rule_registry.get_or_create(rkey, first_rule.confidence)

            if not rejected:
                for p in range(len(draft)):
                    rstats.record_accept(p)
                rule_hash = hash((first_rule.ctx, first_rule.token))
                for p in range(len(draft)):
                    early_exit.record_outcome(rule_hash, p, False)
                adaptive_k.update(len(draft), len(draft))
            else:
                accepted_draft = max(0, accepted_batch - len(pending))
                for p in range(accepted_draft):
                    rstats.record_accept(p)
                    rule_hash = hash((first_rule.ctx, first_rule.token))
                    early_exit.record_outcome(rule_hash, p, False)
                if accepted_draft < len(draft):
                    rstats.record_reject(accepted_draft)
                    rule_hash = hash((first_rule.ctx, first_rule.token))
                    early_exit.record_outcome(rule_hash, accepted_draft, True)
                if accepted_batch == 0:
                    adaptive_k.full_reject_penalty()
                else:
                    adaptive_k.update(accepted_batch, len(draft))

        rule_registry.tick_all()

        proposed += len(draft)
        accepted += accepted_draft
        draft_starts += int(bool(draft))
        accept_hist[accepted_draft] += int(bool(draft))
        reject_count += int(rejected)

        gen = gen[:old_kv] + batch[:accepted_batch] + [bonus]
        kv_len = old_kv + accepted_batch
        model.truncate_kv(kv_len)
        prev_pred = bonus
        if live_ngram is not None:
            live_ngram.feed_accepted(batch[:accepted_batch] + [bonus])

        if live_viz and (draft or rejected):
            reduced = accepted_draft
            total_speed = (len(gen) - len(prompt_ids)) / max(1, passes)
            print(
                f"\r[live] tok={len(gen)-len(prompt_ids):>4}/{max_tokens} "
                f"pass={passes:>4} draft={len(draft)} acc={accepted_draft} "
                f"saved~={reduced:>2} speed_tokens/pass={total_speed:.2f} "
                f"tier={rules[0].tier if rules else 'none':<24}",
                end="",
                flush=True,
            )

        rows.append({
            "kind": "spec_eval",
            "pending": len(pending),
            "draft": len(draft),
            "accepted_batch": accepted_batch,
            "accepted_draft": accepted_draft,
            "bonus": bonus,
            "tier": rules[0].tier if rules else "none",
            "rule_chain": "|".join(r.tier for r in rules),
            "rule_conf": "|".join(f"{r.confidence:.3f}" for r in rules),
            "rule_support": "|".join(str(r.support) for r in rules),
            "pos": pos,
            "ctx_tail_ids": ctx_tail_ids,
            "draft_ids": list(draft),
            "model_draft_ids": model_draft,
            "next_model_id": bonus,
            "mismatch_at": mismatch_at,
            "why": why,
            "rule_strikes": sum(rule_strikes.values()),
            "draft_mistakes": sum(draft_mistakes.values()),
            "pattern_s": pattern_dt,
            "decode_s": decode_dt,
            "verify_s": verify_dt,
        })

    if live_viz:
        print()

    elapsed = now() - start
    out_ids = gen[:target]
    t0 = now()
    text = model.decode(out_ids[len(prompt_ids) :])
    times["detok"] += now() - t0

    if speed_trace is not None:
        speed_trace.finish(
            tokens_generated=len(out_ids) - len(prompt_ids),
            fire_count=draft_starts,
            draft_proposed=proposed,
            draft_accepted=accepted,
            reject_count=reject_count,
            reject_time=reject_time_total,
            verify_time=verify_time_total,
            pattern_time=pattern_time_total,
            passes=passes,
            spec_time=elapsed,
            k_final=adaptive_k.k,
            rule_stats=rule_registry.summary(),
        )

    return out_ids, text, {
        "time": elapsed,
        "passes": passes,
        "target_tokens_evaluated": sum(r["pending"] + r["draft"] for r in rows),
        "proposed": proposed,
        "accepted": accepted,
        "hit_rate": accepted / proposed if proposed else 0.0,
        "draft_starts": draft_starts,
        "reject_count": reject_count,
        "greedy_emitted": greedy_emitted,
        "accept_hist": dict(sorted(accept_hist.items())),
        "times": times,
        "records": rows,
    }


def _propose_from_draft_model(
    draft_model: FastGreedyLlama,
    tokens: list[int],
    max_k: int,
    mistakes: Counter[tuple[tuple[int, ...], int]],
    mistake_limit: int,
    ctx_len: int = 8,
) -> tuple[list[int], list]:
    from .engine import Rule

    if max_k <= 0:
        return [], []
    draft_model.reset()
    logits = draft_model.decode_logits(tokens, logits_all=False)[0]
    tmp = list(tokens)
    draft: list[int] = []
    rules: list[Rule] = []
    for _ in range(max_k):
        tok = draft_model.argmax(logits)
        ctx = tuple(tmp[-min(ctx_len, len(tmp)) :])
        if mistakes[(ctx, tok)] >= mistake_limit:
            break
        draft.append(tok)
        rules.append(Rule(ctx, tok, 0.50, 1, 1, "draft_model"))
        tmp.append(tok)
        if tok == draft_model.eos:
            break
        logits = draft_model.decode_logits([tok], logits_all=False)[0]
    return draft, rules


def observe_qwen_token_shapes(model: FastGreedyLlama, prompts: list[str], tokens: int) -> None:
    print(f"\nQwen live token observation: first {tokens} generated tokens per prompt")
    ngram_counts = Counter()
    for idx, prompt in enumerate(prompts, 1):
        ids, text, _ = run_greedy(model, prompt, tokens)
        prompt_len = len(model.encode(prompt))
        gen = ids[prompt_len : prompt_len + tokens]
        pieces = [model.piece(t).replace("\n", "\\n") for t in gen]
        print(f"\n[{idx:02d}] {prompt[:64]}")
        for pos, (tid, piece) in enumerate(zip(gen, pieces), 1):
            print(f"  {pos:02d} {tid:>7} {piece!r}")
        for n in (2, 3, 4):
            for i in range(0, max(0, len(gen) - n + 1)):
                ngram_counts[tuple(gen[i : i + n])] += 1

    print("\nMost common live token clusters")
    for gram, count in ngram_counts.most_common(30):
        text = model.decode(list(gram)).replace("\n", "\\n")
        ids = " ".join(str(x) for x in gram)
        print(f"  {count:>3}x  [{ids:<24}]  {text!r}")
