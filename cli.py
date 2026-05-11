"""
Qwen DSA pattern speculative decoder - from scratch.

What this script does:
1. Reads engineering_dsa_tokens.json and mines Qwen token-level next-token rules.
2. Scores those rules offline before touching the model.
3. Runs a correct greedy baseline and a correct pending-bonus speculative decoder.
4. Prints pass reduction, target-token eval reduction, hit rate, timing phases, and
   a culprit summary.

The speculative loop uses the important pending-bonus trick:
    generated text may contain one bonus token that is not in KV yet.
    The next target eval verifies [pending_bonus + draft] in one pass.
That avoids the extra "eval bonus immediately" pass that killed previous speedups.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import collections
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from llama_cpp import Llama, LlamaCache


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_MODEL = (
    r"C:\Users\neera\.lmstudio\models\Qwen\Qwen2.5-7B-Instruct-GGUF"
    r"\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
)
DEFAULT_JSON = r"C:\Users\neera\OneDrive\Desktop\sep\engineering_dsa_tokens.json"


PROMPTS = [
    "implement linked list in python only code no comments",
    "implement fibonacci in python, no comments, only code:",
    "implement BST in python ,only code,no comments ",
    "reverse a list in python,only code,no comments",
    "reverse linked list in python ,no comments ,only code",
    "write function in python to check given number prime or not only code no comments",
    "implement merge sort in python only code no comments ",
    "implement quick sort in python only code no comments",
    "write function to check given string palindrome or not in python only code no comments",
    "implement deletion at end in linked list in python only code no comments",
    "implement insertion at beginning in linked list in python only code no comments ",
    "implement deletion of node at beginning in linked list in python only code no comments",
    "implement Min heap in python only code no comments",
    "implement max heap in python only code no comments ",
    "implement binary tree in python only code no comments",
    "implement BST in python only code no comments",
    "write function for reversing linked list in python only code no comments",
    "write function for checking whether given number is prime or not only code no comments",
    "implement stack using list in python only code no comments",
    "implement queue using list in python only code no comments",
]


EXTRA_CODE_EXAMPLES = {
    "fibonacci_recursive": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
""",
    "fibonacci_iterative": """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
""",
    "is_prime_trial_division": """
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
""",
    "sieve_of_eratosthenes": """
def sieve(n):
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if primes[i]:
            for j in range(i * i, n + 1, i):
                primes[j] = False
    return [i for i in range(n + 1) if primes[i]]
""",
    "is_palindrome_string": """
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
""",
    "reverse_list_iterative": """
def reverse_list(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
    return arr
""",
    "reverse_list_recursive": """
def reverse_list(arr):
    if len(arr) <= 1:
        return arr
    return reverse_list(arr[1:]) + [arr[0]]
""",
    "binary_search_recursive": """
def binary_search(arr, target, left, right):
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    if arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    return binary_search(arr, target, mid + 1, right)
""",
}


@dataclass(frozen=True)
class Rule:
    ctx: tuple[int, ...]
    token: int
    confidence: float
    support: int
    total: int
    tier: str


# ─────────────────────────────────────────────
# METHOD 3 — RuleStats + Registry (per-rule cooldown/pre-reject)
# ─────────────────────────────────────────────
class RuleStats:
    __slots__ = ("name", "offline_conf", "fires", "accepts", "rejects",
                 "consecutive_rejects", "last_five", "cooldown_remaining")
    def __init__(self, name: str, offline_conf: float):
        self.name = name
        self.offline_conf = offline_conf
        self.fires = 0
        self.accepts = 0
        self.rejects = 0
        self.consecutive_rejects = 0
        self.last_five: list[int] = []
        self.cooldown_remaining = 0

    @property
    def live_acceptance_rate(self) -> float:
        if len(self.last_five) < 3:
            return self.offline_conf
        return sum(self.last_five) / len(self.last_five)

    def should_fire(self) -> bool:
        if self.cooldown_remaining > 0: return False
        if self.consecutive_rejects >= 5: return False
        if self.live_acceptance_rate < 0.55: return False
        return True

    def record_accept(self, pos: int):
        self.fires += 1; self.accepts += 1; self.consecutive_rejects = 0
        self.last_five.append(1)
        if len(self.last_five) > 5: self.last_five.pop(0)

    def record_reject(self, pos: int):
        self.fires += 1; self.rejects += 1; self.consecutive_rejects += 1
        self.last_five.append(0)
        if len(self.last_five) > 5: self.last_five.pop(0)
        if self.consecutive_rejects >= 5:
            self.cooldown_remaining = 3

    def tick_cooldown(self):
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1


class RuleStatsRegistry:
    def __init__(self):
        self._reg: dict[str, RuleStats] = {}
    def get_or_create(self, name: str, conf: float) -> RuleStats:
        if name not in self._reg:
            self._reg[name] = RuleStats(name, conf)
        return self._reg[name]
    def tick_all(self):
        for s in self._reg.values(): s.tick_cooldown()
    def summary(self):
        return {n: {"fires": s.fires, "accepts": s.accepts, "rejects": s.rejects,
                    "accept_rate": s.live_acceptance_rate, "cooldown": s.cooldown_remaining}
                for n, s in self._reg.items()}


# ─────────────────────────────────────────────
# METHOD 8 — Adaptive K Controller
# ─────────────────────────────────────────────
class AdaptiveKController:
    def __init__(self, k_init=4, k_min=1, k_max=12, window=10,
                 up_thresh=0.93, down_thresh=0.80):
        self.k = k_init; self.k_min = k_min; self.k_max = k_max
        self.window = collections.deque(maxlen=window)
        self.up_thresh = up_thresh; self.down_thresh = down_thresh
    def get_k(self) -> int: return self.k
    def update(self, accepted: int, proposed: int):
        if proposed == 0: return
        rate = accepted / proposed
        self.window.append(rate)
        if len(self.window) < 3: return
        avg = sum(self.window) / len(self.window)
        if avg > self.up_thresh:
            self.k = min(self.k + 1, self.k_max)
        elif avg < self.down_thresh:
            self.k = max(self.k - 1, self.k_min)
    def full_reject_penalty(self):
        self.k = max(self.k_min, self.k - 2)


# ─────────────────────────────────────────────
# METHOD 6 — LastTargetTopKFilter (pre-verify guidance)
# ─────────────────────────────────────────────
class LastTargetTopKFilter:
    def __init__(self, top_k=5):
        self.top_k = top_k
        self._top_ids: set[int] = set()
        self._ready = False
    def update(self, last_logits_row: np.ndarray):
        # last_logits_row: shape (vocab_size,)
        top = np.argpartition(-last_logits_row, self.top_k)[:self.top_k]
        self._top_ids = set(top.tolist())
        self._ready = True
    def first_token_ok(self, token_id: int) -> bool:
        if not self._ready: return True
        return token_id in self._top_ids


# ─────────────────────────────────────────────
# LIVE LOCAL N-GRAM MINER — conservative, only proven continuations
# ─────────────────────────────────────────────
class LiveNgramMiner:
    """Tracks only n-gram continuations that were VERIFIED accepted by target model.
    Only proposes when the same suffix had the SAME continuation before."""

    def __init__(self, max_n: int = 4, window: int = 128):
        self.max_n = max_n
        self.window = window
        self._history: list[int] = []
        # gram -> Counter of next tokens (only from verified accepts)
        self._next_counts: dict[tuple[int, ...], dict[int, int]] = {}

    def feed_accepted(self, tokens: list[int]):
        """Feed a sequence of tokens that were ACCEPTED by target model."""
        self._history.extend(tokens)
        if len(self._history) > self.window:
            self._history = self._history[-self.window:]
        # Index verified continuations
        for i in range(len(tokens)):
            abs_i = len(self._history) - len(tokens) + i
            if abs_i == 0:
                continue
            for n in range(1, self.max_n + 1):
                if abs_i < n:
                    continue
                gram = tuple(self._history[abs_i - n:abs_i])
                nxt = self._history[abs_i]
                self._next_counts.setdefault(gram, {})
                self._next_counts[gram][nxt] = self._next_counts[gram].get(nxt, 0) + 1

    def propose(self, context: list[int], max_k: int) -> list[int] | None:
        """Only propose if every step has a single dominant next token (>80%)."""
        if not self._history or max_k <= 0:
            return None
        result: list[int] = []
        tmp = list(context)
        for _ in range(min(max_k, 2)):  # cap at 2 to limit rejection cost
            gram = None
            for n in range(self.max_n, 0, -1):
                if len(tmp) >= n:
                    g = tuple(tmp[-n:])
                    if g in self._next_counts:
                        gram = g
                        break
            if gram is None:
                break
            counts = self._next_counts[gram]
            total = sum(counts.values())
            best_tok, best_cnt = max(counts.items(), key=lambda x: x[1])
            if best_cnt / total < 0.80:  # not deterministic enough
                break
            result.append(best_tok)
            tmp.append(best_tok)
        return result if result else None


# ─────────────────────────────────────────────
# METHOD 2 — EarlyExitDraftController
# ─────────────────────────────────────────────
class EarlyExitDraftController:
    def __init__(self, min_accept=0.80, min_conf=0.88, risk_thresh=0.30):
        self.min_accept = min_accept
        self.min_conf = min_conf
        self.risk_thresh = risk_thresh
        self.pos_risk: dict[tuple[int, int], tuple[int, int]] = {}  # (hash,pos) -> (rejects, fires)
    def should_extend(self, pat_hash: int, conf: float, accept_rate: float, pos: int) -> bool:
        if accept_rate < self.min_accept: return False
        if pos >= 2 and conf < self.min_conf: return False
        key = (pat_hash, pos)
        if key in self.pos_risk:
            rej, fires = self.pos_risk[key]
            if fires > 5 and (rej / fires) > self.risk_thresh:
                return False
        return True
    def record_outcome(self, pat_hash: int, pos: int, rejected: bool):
        key = (pat_hash, pos)
        rej, fires = self.pos_risk.get(key, (0, 0))
        self.pos_risk[key] = (rej + (1 if rejected else 0), fires + 1)


# ─────────────────────────────────────────────
# METHOD 7 — EntropyProxyAbort
# ─────────────────────────────────────────────
class EntropyProxyAbort:
    def __init__(self, max_rules=5, min_best_conf=0.85):
        self.max_rules = max_rules
        self.min_best_conf = min_best_conf
    def assess(self, rules: list) -> str:
        if not rules: return "greedy"
        if len(rules) > self.max_rules: return "short_k"
        best = max(r.confidence for r in rules)
        return "short_k" if best < self.min_best_conf else "full_k"


# ─────────────────────────────────────────────
# METHOD 1 — RecoveryModeSelector (A/B)
# ─────────────────────────────────────────────
class RecoveryModeSelector:
    def __init__(self, calibration=40):
        self.cal = calibration
        self.stats = {"truncate": {"time":0.0, "tokens":0, "rounds":0},
                      "seq_bonus": {"time":0.0, "tokens":0, "rounds":0}}
        self.round = 0; self.locked = None
    def get_mode(self) -> str:
        return self.locked if self.locked else ("truncate" if self.round % 2 == 0 else "seq_bonus")
    def record(self, mode: str, accepted_tokens: int, elapsed: float):
        if self.locked: return
        s = self.stats[mode]
        s["time"] += elapsed; s["tokens"] += max(accepted_tokens, 1); s["rounds"] += 1
        self.round += 1
        if self.round >= self.cal and not self.locked:
            scores = {m: s["time"]/s["tokens"] for m,s in self.stats.items() if s["tokens"]>0}
            if scores:
                self.locked = min(scores, key=scores.get)
                print(f"[Recovery] locked={self.locked}  scores: { {k:f'{v*1000:.2f}ms/tok' for k,v in scores.items()} }")


# ─────────────────────────────────────────────
# METHOD 5 — GreedyCorrectionSampler
# ─────────────────────────────────────────────
class GreedyCorrectionSampler:
    @staticmethod
    def batch_verify(target_logits: np.ndarray, draft_ids: list[int], ctx_len: int):
        """Return (accepted_count, reject_pos, correction_token)."""
        num_draft = len(draft_ids)
        preds = np.argmax(target_logits[ctx_len: ctx_len+num_draft], axis=1)
        for i, (pred, draft) in enumerate(zip(preds, draft_ids)):
            if int(pred) != draft:
                return i, i, int(pred)  # accepted=i, reject_pos=i, correction=pred
        # all accepted
        bonus = int(np.argmax(target_logits[ctx_len + num_draft]))
        return num_draft, -1, bonus


# ─────────────────────────────────────────────
# SPEED TRACE COLLECTOR
# ─────────────────────────────────────────────
class SpeedTraceCollector:
    """Writes one JSON line per prompt to speed_trace.jsonl."""
    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "a", encoding="utf-8")
        self.current: dict | None = None

    def start(self, prompt_id: int, prompt_text: str, greedy_time: float, greedy_passes: int):
        self.current = {
            "prompt_id": prompt_id,
            "prompt_name": prompt_text[:60],
            "greedy_time": greedy_time,
            "greedy_passes": greedy_passes,
            # fields filled at finish:
            "spec_time": 0.0,
            "spec_passes": 0,
            "tokens_generated": 0,
            "fire_count": 0,
            "draft_proposed": 0,
            "draft_accepted": 0,
            "reject_count": 0,
            "reject_recover_time": 0.0,
            "verify_time": 0.0,
            "pattern_time": 0.0,
            "k_final": 0,
            "rule_stats": [],
        }

    def finish(self, **kwargs):
        if self.current is None:
            return
        self.current.update(kwargs)
        self._file.write(json.dumps(self.current) + "\n")
        self._file.flush()
        self.current = None

    def close(self):
        self._file.close()


def now() -> float:
    return time.perf_counter()


class QwenTokenCorpus:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.items = []
        self.token_text: dict[int, str] = {}
        for name, item in raw.items():
            ids = [int(t["id"]) for t in item["tokens"]]
            toks = [str(t["token"]) for t in item["tokens"]]
            for tid, piece in zip(ids, toks):
                self.token_text[tid] = piece
            self.items.append({"name": name, "ids": ids, "code": item.get("code", "")})

    @property
    def sequences(self) -> list[list[int]]:
        return [x["ids"] for x in self.items]

    def token_name(self, tid: int) -> str:
        return self.token_text.get(tid, f"<{tid}>")

    def print_summary(self, top_n: int = 18) -> None:
        lengths = [len(x["ids"]) for x in self.items]
        counts = Counter(t for seq in self.sequences for t in seq)
        print("\nJSON/Qwen token corpus")
        print(f"  file          : {self.path}")
        print(f"  examples      : {len(self.items)}")
        print(f"  total tokens  : {sum(lengths)}")
        print(f"  length min/max: {min(lengths)} / {max(lengths)}")
        print(f"  unique ids    : {len(counts)}")
        print("  top tokens    :")
        for tid, c in counts.most_common(top_n):
            print(f"    {tid:>6} {self.token_name(tid)!r:<18} {c:>4}")


class PatternMiner:
    def __init__(
        self,
        token_text: dict[int, str],
        max_ctx: int = 8,
        min_support: int = 2,
        min_conf: float = 0.78,
        det_conf: float = 0.96,
        min_rule_ctx: int = 2,
    ):
        self.token_text = token_text
        self.max_ctx = max_ctx
        self.min_support = min_support
        self.min_conf = min_conf
        self.det_conf = det_conf
        self.min_rule_ctx = min_rule_ctx
        self.rules_by_ctx: dict[tuple[int, ...], Rule] = {}
        self.rules_by_len: dict[int, dict[tuple[int, ...], Rule]] = {}
        self.rule_count_by_tier = Counter()

    def fit(self, sequences: Iterable[list[int]]) -> "PatternMiner":
        counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        occurrences: Counter[tuple[int, ...]] = Counter()
        for seq in sequences:
            for i in range(1, len(seq) + 1):
                upto = min(self.max_ctx, i)
                for n in range(1, upto + 1):
                    ctx = tuple(seq[i - n:i])
                    occurrences[ctx] += 1
                    if i < len(seq):
                        counts[ctx][seq[i]] += 1

        rules: dict[tuple[int, ...], Rule] = {}
        for ctx, ctr in counts.items():
            if len(ctx) < self.min_rule_ctx:
                continue
            token, support = ctr.most_common(1)[0]
            total = occurrences[ctx]
            conf = support / total
            if support < self.min_support or conf < self.min_conf:
                continue
            tier = self._tier(ctx, conf, support)
            rules[ctx] = Rule(ctx, token, conf, support, total, tier)

        self.rules_by_ctx = rules
        by_len: dict[int, dict[tuple[int, ...], Rule]] = defaultdict(dict)
        for ctx, rule in rules.items():
            by_len[len(ctx)][ctx] = rule
            self.rule_count_by_tier[rule.tier] += 1
        self.rules_by_len = dict(by_len)
        return self

    def _tier(self, ctx: tuple[int, ...], conf: float, support: int) -> str:
        if conf >= 0.999 and support >= 2:
            return f"det_ctx{len(ctx)}"
        if conf >= self.det_conf:
            return f"strong_ctx{len(ctx)}"
        return f"freq_ctx{len(ctx)}"

    @staticmethod
    def rule_key(rule: Rule) -> tuple[tuple[int, ...], int, str]:
        return (rule.ctx, rule.token, rule.tier)

    def find_rule(self, tokens: list[int], banned: set | dict | None = None) -> Rule | None:
        upto = min(self.max_ctx, len(tokens))
        tail = tokens[-self.max_ctx:] if len(tokens) >= self.max_ctx else tokens
        tail_len = len(tail)
        for n in range(upto, 0, -1):
            bucket = self.rules_by_len.get(n)
            if bucket is None:
                continue
            ctx = tuple(tail[tail_len - n:]) if tail_len >= n else tuple(tokens[-n:])
            rule = bucket.get(ctx)
            if rule is None:
                continue
            if rule.tier == "det_ctx6" and rule.support < 10:
                continue
            if banned is None or self.rule_key(rule) not in banned:
                return rule
        return None

    # Per-tier max draft length caps — low-accuracy tiers get k=1 only
    TIER_K_CAPS: dict[str, int] = {
        "det_ctx5": 1,
        "syntax_while_colon": 1,
        "syntax_class_method_indent": 1,
        "syntax_code_fence_class": 1,
    }

    def propose(
        self,
        tokens: list[int],
        max_k: int,
        banned: set | dict | None = None,
    ) -> tuple[list[int], list[Rule]]:
        draft: list[int] = []
        used: list[Rule] = []
        tmp = list(tokens)
        for step in range(max_k):
            rule = self.find_rule(tmp, banned=banned)
            if rule is None:
                break
            # Tier-specific cap: some rules only ever draft 1 token
            tier_cap = self.TIER_K_CAPS.get(rule.tier, max_k)
            if step >= tier_cap:
                break
            # Allow strong_ctx rules (0.90+) to chain longer; others need det_conf
            if step > 0:
                is_strong = rule.tier.startswith("strong_ctx") and rule.confidence >= 0.90
                if not is_strong and (rule.confidence < self.det_conf or rule.support < 3):
                    break
            draft.append(rule.token)
            used.append(rule)
            tmp.append(rule.token)
        return draft, used

    def offline_next_token_eval(self, sequences: Iterable[list[int]]) -> dict:
        total = predicted = correct = 0
        by_tier = defaultdict(lambda: [0, 0])
        for seq in sequences:
            for i in range(1, len(seq)):
                total += 1
                rule = self.find_rule(seq[:i])
                if rule is None:
                    continue
                predicted += 1
                ok = int(rule.token == seq[i])
                correct += ok
                by_tier[rule.tier][0] += 1
                by_tier[rule.tier][1] += ok
        return {"total": total, "predicted": predicted, "correct": correct, "by_tier": by_tier}

    def offline_chain_eval(self, sequences: Iterable[list[int]], max_k: int) -> dict:
        starts = proposed = accepted = full = 0
        hist = Counter()
        for seq in sequences:
            for i in range(1, len(seq) - 1):
                draft, _ = self.propose(seq[:i], min(max_k, len(seq) - i))
                if not draft:
                    continue
                starts += 1
                proposed += len(draft)
                acc = 0
                for j, tok in enumerate(draft):
                    if i + j < len(seq) and seq[i + j] == tok:
                        acc += 1
                    else:
                        break
                accepted += acc
                full += int(acc == len(draft))
                hist[acc] += 1
        return {
            "starts": starts,
            "proposed": proposed,
            "accepted": accepted,
            "full": full,
            "hist": hist,
        }

    def print_report(self, corpus: QwenTokenCorpus, max_k: int, top_n: int = 25) -> None:
        print("\nMined pattern rules")
        print(f"  max context    : {self.max_ctx}")
        print(f"  min rule ctx   : {self.min_rule_ctx}")
        print(f"  min support    : {self.min_support}")
        print(f"  min confidence : {self.min_conf:.2f}")
        print(f"  rules total    : {len(self.rules_by_ctx)}")
        for tier, c in sorted(self.rule_count_by_tier.items()):
            print(f"    {tier:<12}: {c}")

        nt = self.offline_next_token_eval(corpus.sequences)
        cov = nt["predicted"] / max(1, nt["total"])
        acc = nt["correct"] / max(1, nt["predicted"])
        print("\nOffline next-token test on JSON corpus")
        print(f"  positions      : {nt['total']}")
        print(f"  predictions    : {nt['predicted']} ({cov*100:.1f}% coverage)")
        print(f"  correct        : {nt['correct']} ({acc*100:.1f}% hit rate)")

        ch = self.offline_chain_eval(corpus.sequences, max_k=max_k)
        chain_hit = ch["accepted"] / max(1, ch["proposed"])
        full_rate = ch["full"] / max(1, ch["starts"])
        print("\nOffline chain/draft test")
        print(f"  draft starts   : {ch['starts']}")
        print(f"  proposed       : {ch['proposed']}")
        print(f"  accepted       : {ch['accepted']} ({chain_hit*100:.1f}% token hit)")
        print(f"  full accepted  : {ch['full']} ({full_rate*100:.1f}% of drafts)")
        print(f"  accept histogram: {dict(sorted(ch['hist'].items()))}")

        print("\nTop deterministic-looking rules")
        rules = sorted(
            self.rules_by_ctx.values(),
            key=lambda r: (-r.confidence, -r.support, -len(r.ctx)),
        )[:top_n]
        for r in rules:
            ctx_txt = " ".join(repr(corpus.token_name(x)) for x in r.ctx[-4:])
            print(
                f"  {r.tier:<12} conf={r.confidence:>5.2f} "
                f"sup={r.support:>3}/{r.total:<3}  {ctx_txt} -> "
                f"{r.token} {corpus.token_name(r.token)!r}"
            )


class PythonSyntaxProposer:
    """Tiny Qwen-token syntax backoff for common Python code moves."""

    # Pre-compiled regexes — avoid recompilation on every token step
    _RE_FOR_IN = re.compile(r"^\s*for\s+[A-Za-z_][A-Za-z0-9_]*$")
    _RE_FOR_RANGE = re.compile(r"^\s*for\s+[A-Za-z_][A-Za-z0-9_]*\s+in$")
    _RE_RANGE_PAREN = re.compile(r"^\s*for\b.*\brange$")
    _RE_WHILE_TRUE = re.compile(r"\bwhile\s+True$")
    _RE_WHILE_VAR = re.compile(r"^\s*while\s+[A-Za-z_]\w*$")
    _RE_IF_NAME = re.compile(r"\bif\s+__name__$")
    _RE_IF_NAME_EQ = re.compile(r"\bif\s+__name__\s+==$")
    _RE_IF_NAME_MAIN = re.compile(r'\bif\s+__name__\s+==\s+"__main__"$')
    _RE_ELIF_COLON = re.compile(r"^\s*elif\b.*\)$")
    _RE_FOR_RANGE_COLON = re.compile(r"^\s*for\b.*range\([^\)]+\)$")
    _RE_N_GUARD = re.compile(r"if\s+n\s*<=\s*1$")
    _RE_RETURN_LEN = re.compile(r"\breturn\s+len$")
    _RE_SUPER = re.compile(r"\bsuper\(\)$")
    _RE_SUPER_INIT = re.compile(r"\bsuper\(\)\.__init__\($")
    _RE_DUNDER_INIT = re.compile(r"\bdef\s+__$")
    _RE_DUNDER_PAREN = re.compile(r"\bdef\s+__init$")
    _RE_DUNDER_SELF = re.compile(r"\bdef\s+__init__\($")
    _RE_DEF_COLON = re.compile(r"^(def|class)\b.*\)[^:]$")
    _RE_FLOOR_DIV = re.compile(r"//\s$")
    _RE_PLUSEQ = re.compile(r"\+=\s$")
    _RE_NONE_GUARD = re.compile(r"^.*\bif\b.*\bis\s+(not\s+)?None$")
    _RE_WITH_AS = re.compile(r"^.*\)\s+as\s+[A-Za-z_][A-Za-z0-9_]*$")
    _RE_FIB = re.compile(r"\(n\s*-\s*1\)\s*\+\s*f\(n\s*-$")
    _RE_MINUS_ONE = re.compile(r"(\[.*-|\brange\(.*-|\bn\s*-)$")
    _RE_CLASS_NAME = re.compile(r"^\s*class\s+\w+$")
    _RE_SELF_NONE_ATTR = re.compile(r"\bself\.(data|key|value|val|left|right|next|prev|head|tail|root|node|parent|size|count|length)\s*=\s*$")
    _RE_SELF_LIST_ATTR = re.compile(r"\bself\.(items|heap|stack|queue|arr|data|memo|cache|freq|dp|res|output|path)\s*=\s*$")
    _RE_IF_NOT_SELF = re.compile(r"\bif\s+not\s+self\.(head|root|left|right|data|items|heap|stack|queue|node|val|value|key|next|prev|tail)\s*:\s*$")
    _RE_NOARG_METHOD = re.compile(r"\.(items|keys|values|reverse|clear|copy|lower|upper|isdigit|isalpha|isalnum|isspace|isupper|islower|title|capitalize|strip|lstrip|rstrip|sort|pop|append|extend|count|index|find|replace|split|join|encode|decode|startswith|endswith|format|zfill|center|ljust|rjust)\($")
    _RE_RETURN_TYPE = re.compile(r"\)\s*->\s*(?:None|bool|int|str|float|list|dict|tuple|set|bytes|Any|List|Dict|Tuple|Set|Optional|Union|Callable|Iterable|Iterator|Generator)$")
    _RE_SLICE_COLON = re.compile(r"\[::-\s*$")
    _RE_SLICE_ONE = re.compile(r"\[::-1$")
    _RE_CLASS_DEF_INIT = re.compile(r"class\s+\w+\s*:\n\s+def\s+$")
    _RE_FUNC_CLOSE = re.compile(r"^\s*def\s+\w+\([A-Za-z_]\w*\)$")
    _RE_ELSE_TRY_FINALLY = re.compile(r"^(else|try|finally)$")
    _RE_RETURN_TERMINAL = re.compile(r"^\s*return\s+(True|False|None)$")
    _RE_INCREMENT_SPACE = re.compile(r"^\s*[ijn]\s*\+=$")
    _RE_INCREMENT_END = re.compile(r"^\s*[ijn]\s*\+=\s*1$")
    _RE_WHILE_COND = re.compile(r"^\s*while\s+[A-Za-z_]\w*$")

    def __init__(self, model: "FastGreedyLlama", mode: str = "cluster"):
        self.model = model
        self.mode = mode
        self.max_ctx = 32
        self.prompt_hint: str = ""  # set by benchmark: "class" or "def" or ""
        self.token_ids = {
            " ": self._single(" "),
            " in": self._single(" in"),
            " range": self._single(" range"),
            "(": self._single("("),
            ")": self._single(")"),
            "\n": self._single("\n"),
            ":\n": self._single(":\n"),
            "class": self._single("class"),
            "def": self._single("def"),
            " def": self._single(" def"),
            "init": self._single("init"),
            "__(": self._single("__("),
            "self": self._single("self"),
            "(self": self._single("(self"),
            " len": self._single(" len"),
            " ==": self._single(" =="),
            "1": self._single("1"),
            "2": self._single("2"),
            " 1": self._single(" 1"),
            " =": self._single(" ="),
            " None": self._single(" None"),
            " True": self._single(" True"),
            " False": self._single(" False"),
            " Exception": self._single(" Exception"),
            " as": self._single(" as"),
            " e": self._single(" e"),
            ".__init__(": self._single(".__init__("),
            ' "__main__"': self._single(' "__main__"'),
            "0": self._single("0"),
            " [": self._single(" ["),
            "]": self._single("]"),
            "]\n": self._single("]\n"),
            "len(": self._single("len("),
            "return": self._single("return"),
            "not": self._single("not"),
            "is": self._single("is"),
            ",": self._single(","),
            " other": self._single(" other"),
            "):\n": self._single("):\n"),
            "):": self._single("):"),
            "[::-": self._single("[::-"),
            " []\n\n": self._single(" []\n\n"),
        }
        self.space_tokens: dict[int, int] = {}
        for n in range(1, 33):
            ids = model.tokenize(" " * n)
            if len(ids) == 1:
                self.space_tokens[n] = ids[0]

    def _single(self, text: str) -> int | None:
        ids = self.model.tokenize(text)
        return ids[0] if len(ids) == 1 else None

    def _text_tail(self, tokens: list[int], n: int = 64) -> str:
        return "".join(self.model.token_piece(t) for t in tokens[-n:])

    @staticmethod
    def _previous_line_indent(text: str) -> int:
        lines = text.split("\n")
        if len(lines) < 2:
            return 0
        prev = lines[-2]
        return len(prev) - len(prev.lstrip(" "))

    def _space_token_for_next_indent(self, text: str) -> int | None:
        # Python standard indent is 4 spaces.
        # Qwen often encodes one space on the following word token,
        # so the pure-space token is desired_indent - 1.
        desired = self._previous_line_indent(text) + 4
        return self.space_tokens.get(desired - 1)

    def _last_nonblank_indent(self, text: str) -> int:
        for line in reversed(text.split("\n")):
            if line.strip():
                return len(line) - len(line.lstrip(" "))
        return 0

    def _body_indent_after_colon(self, text: str) -> int | None:
        """If the last non-empty line ended with ':' and current line is empty,
        predict the proper indentation for the new block body."""
        if not text.endswith("\n"):
            return None
        lines = text.rstrip("\n").split("\n")
        if not lines:
            return None
        # Current line must be empty (just whitespace)
        if lines[-1].strip():
            return None
        # Find last non-empty line
        for line in reversed(lines[:-1]):
            if line.strip():
                if line.rstrip().endswith(":"):
                    indent = len(line) - len(line.lstrip(" "))
                    return self.space_tokens.get(indent + 3)  # +4 indent, -1 for Qwen encoding
                break
        return None

    def _class_next_method_indent(self, text: str) -> int | None:
        if not text.endswith("\n\n"):
            return None
        lines = text.split("\n")
        has_recent_class = any(line.lstrip().startswith("class ") for line in lines[-12:])
        if not has_recent_class:
            return None
        for line in reversed(lines[:-2]):
            stripped = line.strip()
            if not stripped:
                continue
            # A finished method body inside a class usually dedents to the
            # class body level; Qwen then emits one less pure space because
            # the next token is " def" with a leading space.
            indent = len(line) - len(line.lstrip(" "))
            if indent >= 7 and not stripped.startswith(("return", "break", "continue")):
                return 3
            break
        return None

    def find_rule(
        self,
        tokens: list[int],
        banned: set | dict | None = None,
    ) -> Rule | None:
        if not tokens:
            return None
        text = self._text_tail(tokens)
        if text.rstrip().endswith("```"):
            return None

        line = text.split("\n")[-1]
        stripped = line.strip()
        token: int | None = None
        conf = 0.97
        tier = "syntax"

        def choose(piece: str, confidence: float, name: str) -> None:
            nonlocal token, conf, tier
            token = self.token_ids.get(piece)
            conf = confidence
            tier = name

        # ── Class method indent ──
        class_method_indent = self._class_next_method_indent(text)
        if class_method_indent is not None:
            token = self.space_tokens.get(class_method_indent)
            tier = "syntax_class_method_indent"
            conf = 0.99

        # ── Code fence ──
        if token is None and (text.endswith("```python\n") or text.endswith("``` python\n")):
            # Route by prompt hint: class-heavy prompts → class, others → def
            if self.prompt_hint == "class":
                choose("class", 0.97, "syntax_code_fence_class")
            elif self.prompt_hint == "def":
                choose("def", 0.97, "syntax_code_fence_def")
            elif re.search(r"(linked|tree|heap|stack|queue|bst|node|graph|sort|search)", text, re.I):
                choose("class", 0.97, "syntax_code_fence_class")
            else:
                choose("def", 0.97, "syntax_code_fence_def")

        # ── Fast syntax rules (pre-compiled regexes) ──
        if token is None:
            if self._RE_ELSE_TRY_FINALLY.match(stripped):
                choose(":\n", 1.00, "syntax_block_colon")
            elif self._RE_RETURN_TERMINAL.match(line):
                choose("\n", 1.00, "syntax_return_terminal")
            elif stripped in {"break", "continue", "pass"}:
                choose("\n", 1.00, "syntax_statement_end")
            elif self._RE_FOR_IN.match(line):
                choose(" in", 0.99, "syntax_for_in")
            elif self._RE_FOR_RANGE.match(line):
                choose(" range", 0.97, "syntax_for_range")
            elif self._RE_RANGE_PAREN.match(line):
                choose("(", 1.00, "syntax_range_paren")
            elif self._RE_WHILE_TRUE.search(line):
                choose(":\n", 1.00, "syntax_while_true")
            elif self._RE_WHILE_VAR.search(line) and not line.rstrip().endswith(":"):
                choose(":\n", 1.00, "syntax_while_colon")
            elif self._RE_IF_NAME.search(line):
                choose(" ==", 1.00, "syntax_name_eq")
            elif self._RE_IF_NAME_EQ.search(line):
                choose(' "__main__"', 1.00, "syntax_name_main")
            elif self._RE_IF_NAME_MAIN.search(line):
                choose(":\n", 1.00, "syntax_main_colon")
            elif stripped == "try":
                choose(":\n", 1.00, "syntax_try_colon")
            elif stripped == "except":
                choose(" Exception", 0.85, "syntax_except_exception")
            elif stripped == "except Exception":
                choose(" as", 0.90, "syntax_except_as")
            elif stripped == "except Exception as":
                choose(" e", 0.98, "syntax_except_e")
            elif stripped == "except Exception as e":
                choose(":\n", 1.00, "syntax_except_colon")
            elif self._RE_SUPER.search(line):
                choose(".__init__(", 0.98, "syntax_super_init")
            elif self._RE_SUPER_INIT.search(line):
                choose(")", 0.90, "syntax_super_close")
            elif self._RE_DUNDER_INIT.search(line):
                choose("init", 1.00, "syntax_dunder_init")
            elif self._RE_DUNDER_PAREN.search(line):
                choose("__(", 1.00, "syntax_dunder_paren")
            elif self._RE_DUNDER_SELF.search(line):
                choose("self", 1.00, "syntax_dunder_self")
            elif self._RE_RETURN_LEN.search(line):
                choose("(self", 1.00, "syntax_return_len_self")
            elif self._RE_RETURN_TYPE.search(line):
                choose(":\n", 1.00, "syntax_return_annotation_colon")
            elif self._RE_CLASS_NAME.match(line):
                choose(":\n", 1.00, "syntax_class_colon")
            elif self._RE_ELIF_COLON.search(line) and not line.rstrip().endswith(":"):
                choose(":\n", 1.00, "syntax_elif_colon")
            elif self._RE_N_GUARD.search(line):
                choose(":\n", 1.00, "syntax_n_guard_colon")
            elif self._RE_NONE_GUARD.match(line):
                choose(":\n", 0.97, "syntax_none_guard_colon")
            elif self._RE_WITH_AS.match(line):
                choose(":\n", 1.00, "syntax_with_as_colon")
            elif line.rstrip().endswith("//"):
                choose(" ", 1.00, "syntax_floor_div_space")
            elif self._RE_FLOOR_DIV.search(line):
                choose("2", 1.00, "syntax_floor_div_two")
            elif line.rstrip().endswith("+="):
                choose(" ", 1.00, "syntax_pluseq_space")
            elif self._RE_PLUSEQ.search(line):
                choose("1", 1.00, "syntax_pluseq_one")
            elif self._RE_INCREMENT_SPACE.match(line):
                choose(" ", 1.00, "syntax_increment_space")
            elif self._RE_INCREMENT_END.match(line):
                choose("\n", 1.00, "syntax_increment_end")
            elif self._RE_FIB.search(line):
                choose("2", 1.00, "syntax_fib_n_minus_two")
            elif self._RE_MINUS_ONE.search(line):
                choose("1", 0.93, "syntax_minus_one")

            # ── self.attr = init (list attrs only — highest value) ──
            elif self._RE_SELF_LIST_ATTR.search(line):
                choose(" []\n\n", 0.97, "syntax_self_list_init")

            # ── Slice patterns ──
            elif self._RE_SLICE_COLON.search(line):
                choose("1", 1.00, "syntax_slice_one")
            elif self._RE_SLICE_ONE.search(line):
                choose("]\n", 0.98, "syntax_slice_close")

            # ── Fallback: colon-line indent ──
            elif text.endswith(":\n") or text.endswith("):\n") or text.endswith("]:\n"):
                token = self._space_token_for_next_indent(text)
                tier = "syntax_indent"
                conf = 0.98

        if token is None:
            return None
        ctx = tuple(tokens[-min(self.max_ctx, len(tokens)):])
        rule = Rule(ctx, token, conf, 999, 1000, tier)
        if banned is not None and PatternMiner.rule_key(rule) in banned:
            return None
        return rule


def propose_draft(
    miner: PatternMiner,
    syntax: PythonSyntaxProposer | None,
    tokens: list[int],
    max_k: int,
    banned: set | dict | None = None,
    early_exit: EarlyExitDraftController | None = None,
    rule_registry: RuleStatsRegistry | None = None,
    live_ngram: LiveNgramMiner | None = None,
) -> tuple[list[int], list[Rule]]:
    draft: list[int] = []
    used: list[Rule] = []
    tmp = list(tokens)
    for step in range(max_k):
        mined = miner.find_rule(tmp, banned=banned)
        syntactic = syntax.find_rule(tmp, banned=banned) if syntax is not None else None
        if syntactic is not None and (
            mined is None
            or syntactic.confidence >= 0.99
            or syntactic.tier in {"syntax_for_in", "syntax_range_paren", "syntax_indent"}
        ):
            rule = syntactic
        else:
            rule = mined
        if rule is None:
            break
        if step > 0:
            is_high_trust = (
                rule.tier.startswith("syntax_")
                or (rule.tier in {"det_ctx8", "det_ctx7", "det_ctx6"} and rule.confidence >= 0.97)
            )
            if not is_high_trust and (rule.confidence < miner.det_conf or rule.support < 3):
                break
        # Method 2: Early-exit drafting — stop if rule looks risky at this position
        if early_exit is not None and rule_registry is not None and step > 0:
            rstats = rule_registry.get_or_create(f"{rule.tier}:{rule.token}", rule.confidence)
            if not early_exit.should_extend(
                hash((rule.ctx, rule.token)),
                rule.confidence,
                rstats.live_acceptance_rate,
                step,
            ):
                break
        draft.append(rule.token)
        used.append(rule)
        tmp.append(rule.token)
        if rule.tier in {"syntax_floor_div_space", "syntax_floor_div_two", "syntax_pluseq_one"}:
            break
        if syntax is not None and syntax.mode == "basic" and rule.tier.startswith("syntax"):
            break
    # Fallback: live local n-gram mining
    if not draft and live_ngram is not None:
        ngram_draft = live_ngram.propose(tokens, max_k)
        if ngram_draft:
            draft = ngram_draft
            used = [Rule(tuple(tokens[-4:]), t, 0.85, 1, 1, "live_ngram") for t in ngram_draft]
    return draft, used


def propose_from_draft_model(
    draft_model: "FastGreedyLlama",
    tokens: list[int],
    max_k: int,
    mistakes: Counter[tuple[tuple[int, ...], int]],
    mistake_limit: int,
    ctx_len: int = 8,
) -> tuple[list[int], list[Rule]]:
    if max_k <= 0:
        return [], []
    draft_model.reset()
    logits = draft_model.decode_logits(tokens, logits_all=False)[0]
    tmp = list(tokens)
    draft: list[int] = []
    rules: list[Rule] = []
    for _ in range(max_k):
        tok = draft_model.argmax(logits)
        ctx = tuple(tmp[-min(ctx_len, len(tmp)):])
        if mistakes[(ctx, tok)] >= mistake_limit:
            break
        draft.append(tok)
        rules.append(Rule(ctx, tok, 0.50, 1, 1, "draft_model"))
        tmp.append(tok)
        if tok == draft_model.eos:
            break
        logits = draft_model.decode_logits([tok], logits_all=False)[0]
    return draft, rules


class FastGreedyLlama:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1):
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

    def tokenize(self, text: str) -> list[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def detokenize(self, toks: list[int]) -> str:
        if not toks:
            return ""
        return self.llm.detokenize(toks).decode("utf-8", errors="ignore")

    def token_piece(self, tok: int) -> str:
        tok = int(tok)
        piece = self._piece_cache.get(tok)
        if piece is None:
            piece = self.detokenize([tok])
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
            batch = tokens[i:min(len(tokens), i + self.llm.n_batch)]
            n_past = self.llm.n_tokens
            self.llm._batch.set_batch(batch=batch, n_past=n_past, logits_all=logits_all)
            self.llm._ctx.decode(self.llm._batch)
            self.llm.input_ids[n_past:n_past + len(batch)] = batch
            rows = len(batch) if logits_all else 1
            ptr = self.llm._ctx.get_logits()
            out = np.ctypeslib.as_array(ptr, shape=(rows * self.n_vocab,)).reshape(rows, self.n_vocab)
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
    prompt_ids = model.tokenize(prompt)

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
    text = model.detokenize(out_ids[len(prompt_ids):])
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
    # Optimization components
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
    prompt_ids = model.tokenize(prompt)

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
    reject_time_total = 0.0    # separate timing for rejection recovery
    pattern_time_total = 0.0
    verify_time_total = 0.0
    accept_hist = Counter()
    banned_rules: dict[tuple[tuple[int, ...], int, str], int] = {}
    rule_strikes: Counter[tuple[tuple[int, ...], int, str]] = Counter()
    draft_mistakes = draft_mistakes if draft_mistakes is not None else Counter()
    BAN_DECAY = 50

    # Optimization components (create if not provided)
    rule_registry = rule_registry or RuleStatsRegistry()
    adaptive_k = adaptive_k or AdaptiveKController(k_init=k)
    top_k_filter = top_k_filter or LastTargetTopKFilter(top_k=5)
    early_exit = early_exit or EarlyExitDraftController()
    entropy_abort = entropy_abort or EntropyProxyAbort()
    live_ngram = None  # DISABLED: causes more rejections than accepted drafts
    # RecoveryModeSelector can be added later; for now we use fixed reject_mode

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

        # If there is no pending token and no useful draft, emit the model's
        # already-known greedy token as pending. Next loop can draft after it.
        # Method 8: adaptive k controls draft length dynamically
        k_now = adaptive_k.get_k() if adaptive_k is not None else k
        max_draft = max(0, min(k_now, remaining - 1))
        ctx_tail_ids = gen[max(0, len(gen) - 12):]
        pos = len(gen) - len(prompt_ids)

        # Decay banned rules
        expired = [key for key, ban_pos in banned_rules.items() if pos - ban_pos > BAN_DECAY]
        for key in expired:
            del banned_rules[key]

        t0 = now()
        draft, rules = propose_draft(
            miner, syntax, gen, max_draft, banned=banned_rules,
            early_exit=early_exit, rule_registry=rule_registry,
            live_ngram=live_ngram,
        )
        if not draft and draft_model is not None and max_draft > 0:
            draft, rules = propose_from_draft_model(
                draft_model, gen, max_draft, draft_mistakes, draft_mistake_limit
            )
        pattern_dt = now() - t0
        times["pattern"] += pattern_dt
        pattern_time_total += pattern_dt

        # ── METHOD 6 + 3: pre-filter first rule ──
        # NOTE: Method 6 (top-5 filter) disabled — it uses stale logits and kills coverage.
        # Method 3 (rule stats cooldown) kept but applied post-verify.
        if draft and rules:
            first_rule = rules[0]
            # METHOD 3: RuleStats pre-rejection (soft gate)
            rkey = f"{first_rule.tier}:{first_rule.token}"
            rstats = rule_registry.get_or_create(rkey, first_rule.confidence)
            if not rstats.should_fire():
                draft, rules = [], []
                 # else: allow draft to proceed, rstats will be used later

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

        # ── REJECT RECOVERY with timing ──
        rec_time = 0.0
        if rejected and reject_mode == "rebuild":
            accepted_batch = accepted_pending
            accepted_draft = 0
            clean_prefix = gen[:old_kv] + batch[:accepted_batch]
            model.reset()
            t1 = now()
            clean_logits = model.decode_logits(prompt_ids, logits_all=False)[0]
            rebuild_passes = 1
            for replay_tok in clean_prefix[len(prompt_ids):]:
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
            # truncate: nothing extra, time accounted in normal flow
            rec_time = 0.0

        reject_time_total += rec_time

        # ── METHOD 6: update top-K cache (every verify pass) ──
        if top_k_filter is not None:
            top_k_filter.update(batch_logits[-1])

        # ── METHOD 3,2,8: rule stats, early-exit, adaptive k ──
        if draft and rules:
            first_rule = rules[0]
            rkey = f"{first_rule.tier}:{first_rule.token}"
            rstats = rule_registry.get_or_create(rkey, first_rule.confidence)

            if not rejected:
                # full accept
                for pos in range(len(draft)):
                    rstats.record_accept(pos)
                rule_hash = hash((first_rule.ctx, first_rule.token))
                for pos in range(len(draft)):
                    early_exit.record_outcome(rule_hash, pos, False)
                adaptive_k.update(len(draft), len(draft))
            else:
                # partial accept + reject
                accepted_draft = max(0, accepted_batch - len(pending))
                for pos in range(accepted_draft):
                    rstats.record_accept(pos)
                    rule_hash = hash((first_rule.ctx, first_rule.token))
                    early_exit.record_outcome(rule_hash, pos, False)
                if accepted_draft < len(draft):
                    rstats.record_reject(accepted_draft)
                    rule_hash = hash((first_rule.ctx, first_rule.token))
                    early_exit.record_outcome(rule_hash, accepted_draft, True)
                # Adaptive k update
                if accepted_batch == 0:
                    adaptive_k.full_reject_penalty()
                else:
                    adaptive_k.update(accepted_batch, len(draft))

        # Tick cooldowns for all rules (METHOD 3)
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
        # Feed accepted tokens to live n-gram miner (only verified tokens)
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
            "decode_s": decode_dt,
            "verify_s": verify_dt,
        })

    if live_viz:
        print()

    elapsed = now() - start
    out_ids = gen[:target]
    t0 = now()
    text = model.detokenize(out_ids[len(prompt_ids):])
    times["detok"] += now() - t0

    # Write speed trace if enabled
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


TRACE_FIELDS = [
    "prompt", "kind", "pos", "pending", "draft", "accepted_batch",
    "accepted_draft", "bonus", "next_model_id", "tier", "rule_chain",
    "rule_conf", "rule_support", "ctx_tail_ids", "ctx_tail_text",
    "draft_ids", "draft_text", "model_draft_ids", "model_draft_text",
    "next_model_text", "mismatch_at", "why", "rule_strikes", "draft_mistakes",
    "decode_s", "verify_s",
]


def _ids_csv(ids: list[int] | tuple[int, ...] | int | str) -> str:
    if isinstance(ids, (list, tuple)):
        return " ".join(str(int(x)) for x in ids)
    return str(ids)


def _trace_text(model: FastGreedyLlama, ids: list[int] | tuple[int, ...]) -> str:
    if not ids:
        return ""
    return model.detokenize([int(x) for x in ids]).replace("\n", "\\n")


def enrich_trace_records(model: FastGreedyLlama, records: list[dict]) -> list[dict]:
    enriched = []
    for r in records:
        row = dict(r)
        ctx_ids = list(row.get("ctx_tail_ids", []))
        draft_ids = list(row.get("draft_ids", []))
        model_ids = list(row.get("model_draft_ids", []))
        next_id = row.get("next_model_id", row.get("bonus", ""))
        row["ctx_tail_text"] = _trace_text(model, ctx_ids)
        row["draft_text"] = _trace_text(model, draft_ids)
        row["model_draft_text"] = _trace_text(model, model_ids)
        row["next_model_text"] = _trace_text(model, [next_id]) if next_id != "" else ""
        row["ctx_tail_ids"] = _ids_csv(ctx_ids)
        row["draft_ids"] = _ids_csv(draft_ids)
        row["model_draft_ids"] = _ids_csv(model_ids)
        enriched.append(row)
    return enriched


def write_trace_csv(path: Path, prompt: str, records: list[dict], model: FastGreedyLlama | None = None) -> None:
    if not records:
        return
    fields = TRACE_FIELDS
    out_records = enrich_trace_records(model, records) if model is not None else records
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in out_records:
            writer.writerow({"prompt": prompt, **r})


def observe_qwen_token_shapes(model: FastGreedyLlama, prompts: list[str], tokens: int) -> None:
    print(f"\nQwen live token observation: first {tokens} generated tokens per prompt")
    ngram_counts = Counter()
    for idx, prompt in enumerate(prompts, 1):
        ids, text, _ = run_greedy(model, prompt, tokens)
        prompt_len = len(model.tokenize(prompt))
        gen = ids[prompt_len:prompt_len + tokens]
        pieces = [model.token_piece(t).replace("\n", "\\n") for t in gen]
        print(f"\n[{idx:02d}] {prompt[:64]}")
        for pos, (tid, piece) in enumerate(zip(gen, pieces), 1):
            print(f"  {pos:02d} {tid:>7} {piece!r}")
        for n in (2, 3, 4):
            for i in range(0, max(0, len(gen) - n + 1)):
                ngram_counts[tuple(gen[i:i + n])] += 1

    print("\nMost common live token clusters")
    for gram, count in ngram_counts.most_common(30):
        text = model.detokenize(list(gram)).replace("\n", "\\n")
        ids = " ".join(str(x) for x in gram)
        print(f"  {count:>3}x  [{ids:<24}]  {text!r}")


def extra_corpus_sequences(model: FastGreedyLlama) -> list[list[int]]:
    seqs = []
    for code in EXTRA_CODE_EXAMPLES.values():
        text = "\n```python\n" + code.strip() + "\n```"
        seqs.append(model.tokenize(text))
    return seqs


def refit_miner(miner: PatternMiner, sequences: list[list[int]]) -> None:
    miner.rule_count_by_tier = Counter()
    miner.fit(sequences)


def benchmark(args: argparse.Namespace) -> None:
    corpus = QwenTokenCorpus(args.token_json)
    corpus.print_summary()

    miner = PatternMiner(
        corpus.token_text,
        max_ctx=args.max_ctx,
        min_support=args.min_support,
        min_conf=args.min_conf,
        det_conf=args.det_conf,
        min_rule_ctx=args.min_rule_ctx,
    ).fit(corpus.sequences)
    miner.print_report(corpus, max_k=args.k)

    if args.offline_only:
        return

    print(f"\nLoading model: {args.model}")
    model = FastGreedyLlama(args.model, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers)
    prompts = PROMPTS[:args.prompts]
    if args.observe_tokens:
        observe_qwen_token_shapes(model, prompts, args.observe_tokens)
        if args.observe_only:
            return

    syntax_mode = "off" if args.no_syntax_patterns else args.syntax_mode
    syntax = None if syntax_mode == "off" else PythonSyntaxProposer(model, mode=syntax_mode)
    reject_mode = "truncate" if args.unsafe_fast_reject else args.reject_mode
    draft_model = None
    if args.draft_model:
        print(f"Loading draft model: {args.draft_model}")
        draft_model = FastGreedyLlama(args.draft_model, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers)
    draft_mistakes: Counter[tuple[tuple[int, ...], int]] = Counter()

    train_sequences = list(corpus.sequences)
    if args.extra_corpus:
        extra = extra_corpus_sequences(model)
        train_sequences.extend(extra)
        refit_miner(miner, train_sequences)
        print(f"\nAugmented corpus: +{len(extra)} missing DSA examples, rules={len(miner.rules_by_ctx)}")

    agg = Counter()
    time_g = time_s = 0.0
    pass_g = pass_s = 0
    eval_g = eval_s = 0
    ok = 0
    all_records = []

    print("\nCorrect pending-bonus speculative benchmark")
    print(f"  K={args.k}  max_tokens={args.tokens}  prompts={len(prompts)}")
    print("  NOTE: pass = actual target decode call. Bonus is NOT eval'd separately.\n")

    for idx, prompt in enumerate(prompts, 1):
        g_ids, g_text, gs = run_greedy(model, prompt, args.tokens)
        prompt_ids_for_mining = model.tokenize(prompt)
        if args.mine_greedy_before_spec:
            train_sequences.append(g_ids[len(prompt_ids_for_mining):])
            refit_miner(miner, train_sequences)
        s_ids, s_text, ss = run_speculative(
            model, miner, syntax, prompt, args.tokens, args.k,
            reject_mode=reject_mode,
            strike_limit=args.strike_limit,
            draft_model=draft_model,
            draft_mistakes=draft_mistakes,
            draft_mistake_limit=args.draft_mistake_limit,
            live_viz=args.live_viz,
        )
        if args.live_mining:
            train_sequences.append(s_ids[len(prompt_ids_for_mining):])
            refit_miner(miner, train_sequences)
        identical = g_ids == s_ids
        ok += int(identical)

        time_g += gs["time"]; time_s += ss["time"]
        pass_g += gs["passes"]; pass_s += ss["passes"]
        eval_g += gs["target_tokens_evaluated"]; eval_s += ss["target_tokens_evaluated"]
        for name, val in ss["times"].items():
            agg[f"spec_{name}"] += val
        for name, val in gs["times"].items():
            agg[f"greedy_{name}"] += val
        all_records.extend({"prompt": prompt, **r} for r in ss["records"])

        print(f"[{idx:02d}/{len(prompts)}] {prompt[:62]}")
        print(f"  passes     greedy={gs['passes']:>4}  spec={ss['passes']:>4}  "
              f"speed={gs['passes']/max(1, ss['passes']):.3f}x")
        print(f"  eval toks   greedy={gs['target_tokens_evaluated']:>4}  "
              f"spec={ss['target_tokens_evaluated']:>4}  "
              f"ratio={gs['target_tokens_evaluated']/max(1, ss['target_tokens_evaluated']):.3f}x")
        print(f"  time        greedy={gs['time']:.3f}s  spec={ss['time']:.3f}s  "
              f"speed={gs['time']/max(1e-9, ss['time']):.3f}x")
        print(f"  drafts      accepted={ss['accepted']}/{ss['proposed']} "
              f"({ss['hit_rate']*100:.1f}%)  starts={ss['draft_starts']} "
              f"rejects={ss['reject_count']}  greedy_pending={ss['greedy_emitted']}")
        print(f"  match       {'YES' if identical else 'NO'}")
        if not identical:
            diff = next((i for i, (a, b) in enumerate(zip(g_ids, s_ids)) if a != b), None)
            print(f"  first diff  token_index={diff} greedy={g_ids[diff] if diff is not None else None} "
                  f"spec={s_ids[diff] if diff is not None else None}")
        print(f"  spec time   pattern={ss['times']['pattern']:.5f}s "
              f"decode={ss['times']['decode']:.3f}s verify={ss['times']['verify']:.5f}s\n")

    print("=" * 78)
    print("AGGREGATE")
    print(f"  identical outputs : {ok}/{len(prompts)}")
    print(f"  passes            : greedy={pass_g} spec={pass_s} "
          f"speed={pass_g/max(1, pass_s):.3f}x")
    print(f"  target eval tokens: greedy={eval_g} spec={eval_s} "
          f"ratio={eval_g/max(1, eval_s):.3f}x")
    print(f"  wall time         : greedy={time_g:.3f}s spec={time_s:.3f}s "
          f"speed={time_g/max(1e-9, time_s):.3f}x")
    if all_records:
        draft_rows = sum(1 for r in all_records if r["draft"] > 0)
        proposed_total = sum(int(r["draft"]) for r in all_records)
        accepted_total = sum(int(r["accepted_draft"]) for r in all_records)
        reject_total = sum(1 for r in all_records if r["why"] == "draft_token_mismatch")
        no_rule_total = sum(1 for r in all_records if r["why"] == "no_pattern_fired")
        print("\n  Draft coverage")
        print(f"  pattern fire rate : {draft_rows}/{len(all_records)} "
              f"({draft_rows/max(1, len(all_records))*100:.1f}%)")
        print(f"  draft tokens      : accepted={accepted_total}/{proposed_total} "
              f"({accepted_total/max(1, proposed_total)*100:.1f}%)")
        print(f"  rejected drafts   : {reject_total}")
        print(f"  no-rule passes    : {no_rule_total}")
    print("\n  Time culprit table")
    print("  phase          greedy(s)    spec(s)")
    for phase in ("prompt", "pattern", "decode", "verify", "detok"):
        print(f"  {phase:<12} {agg[f'greedy_{phase}']:>10.4f} {agg[f'spec_{phase}']:>10.4f}")

    trace_path = Path(args.trace_csv)
    if all_records:
        fields = TRACE_FIELDS
        out_records = enrich_trace_records(model, all_records)
        with trace_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(out_records)
        print(f"\n  wrote pass trace: {trace_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--token-json", default=DEFAULT_JSON)
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-ctx", type=int, default=8)
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--min-conf", type=float, default=0.96)
    ap.add_argument("--det-conf", type=float, default=0.96)
    ap.add_argument("--min-rule-ctx", type=int, default=4)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--offline-only", action="store_true")
    ap.add_argument("--no-syntax-patterns", action="store_true",
                    help="Disable Python syntax backoff rules such as for->in and colon->indent.")
    ap.add_argument("--syntax-mode", choices=["off", "basic", "cluster"], default="cluster",
                    help="basic predicts one syntax token; cluster chains high-confidence syntax tokens.")
    ap.add_argument("--observe-tokens", type=int, default=0,
                    help="Print live Qwen token ids/pieces for the first N generated tokens per prompt.")
    ap.add_argument("--observe-only", action="store_true")
    ap.add_argument("--extra-corpus", action=argparse.BooleanOptionalAction, default=True,
                    help="Add built-in fibonacci/prime/palindrome/reverse-list examples to the miner.")
    ap.add_argument("--live-mining", action="store_true",
                    help="Mine completed spec outputs and refit for later prompts in this run.")
    ap.add_argument("--mine-greedy-before-spec", action="store_true",
                    help="Oracle/warmup mode: mine each greedy baseline before speculative run for ceiling tests.")
    ap.add_argument("--live-viz", action="store_true",
                    help="Stream accept/reject and approximate pass savings while decoding.")
    ap.add_argument("--draft-model", default="",
                    help="Optional draft model GGUF path. Rules fire first; draft model fills only no-rule gaps.")
    ap.add_argument("--draft-mistake-limit", type=int, default=1,
                    help="In draft model mode, never repeat a context/token mistake after this many failures.")
    ap.add_argument("--strike-limit", type=int, default=3,
                    help="Ban a bad rule only after this many mismatches in one prompt.")
    ap.add_argument("--reject-mode", choices=["truncate", "seq-bonus", "rebuild"], default="seq-bonus",
                    help="seq-bonus cheaply recomputes bonus after rejection; rebuild replays KV after rejection.")
    ap.add_argument("--unsafe-fast-reject", action="store_true",
                    help="Deprecated alias for --reject-mode truncate.")
    ap.add_argument("--trace-csv", default=r"C:\Users\neera\OneDrive\Desktop\sep\qwen_spec_trace.csv")
    return ap.parse_args()


def main() -> None:
    benchmark(parse_args())


if __name__ == "__main__":
    main()
