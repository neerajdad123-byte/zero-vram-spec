from __future__ import annotations

import collections
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal tokenizer interface for speculative decoding."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...
    def piece(self, token_id: int) -> str: ...


@dataclass(frozen=True)
class Rule:
    ctx: tuple[int, ...]
    token: int
    confidence: float
    support: int
    total: int
    tier: str


class RuleStats:
    __slots__ = (
        "name",
        "offline_conf",
        "fires",
        "accepts",
        "rejects",
        "consecutive_rejects",
        "last_five",
        "cooldown_remaining",
    )

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
        if self.cooldown_remaining > 0:
            return False
        if self.consecutive_rejects >= 5:
            return False
        if self.live_acceptance_rate < 0.55:
            return False
        return True

    def record_accept(self, pos: int):
        self.fires += 1
        self.accepts += 1
        self.consecutive_rejects = 0
        self.last_five.append(1)
        if len(self.last_five) > 5:
            self.last_five.pop(0)

    def record_reject(self, pos: int):
        self.fires += 1
        self.rejects += 1
        self.consecutive_rejects += 1
        self.last_five.append(0)
        if len(self.last_five) > 5:
            self.last_five.pop(0)
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
        for s in self._reg.values():
            s.tick_cooldown()

    def summary(self):
        return {
            n: {
                "fires": s.fires,
                "accepts": s.accepts,
                "rejects": s.rejects,
                "accept_rate": s.live_acceptance_rate,
                "cooldown": s.cooldown_remaining,
            }
            for n, s in self._reg.items()
        }


class AdaptiveKController:
    def __init__(
        self,
        k_init: int = 4,
        k_min: int = 1,
        k_max: int = 12,
        window: int = 10,
        up_thresh: float = 0.93,
        down_thresh: float = 0.80,
    ):
        self.k = k_init
        self.k_min = k_min
        self.k_max = k_max
        self.window = collections.deque(maxlen=window)
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh

    def get_k(self) -> int:
        return self.k

    def update(self, accepted: int, proposed: int):
        if proposed == 0:
            return
        rate = accepted / proposed
        self.window.append(rate)
        if len(self.window) < 3:
            return
        avg = sum(self.window) / len(self.window)
        if avg > self.up_thresh:
            self.k = min(self.k + 1, self.k_max)
        elif avg < self.down_thresh:
            self.k = max(self.k - 1, self.k_min)

    def full_reject_penalty(self):
        self.k = max(self.k_min, self.k - 2)


class EarlyExitDraftController:
    def __init__(self, min_accept: float = 0.80, min_conf: float = 0.88, risk_thresh: float = 0.30):
        self.min_accept = min_accept
        self.min_conf = min_conf
        self.risk_thresh = risk_thresh
        self.pos_risk: dict[tuple[int, int], tuple[int, int]] = {}

    def should_extend(self, pat_hash: int, conf: float, accept_rate: float, pos: int) -> bool:
        if accept_rate < self.min_accept:
            return False
        if pos >= 2 and conf < self.min_conf:
            return False
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


class EntropyProxyAbort:
    def __init__(self, max_rules: int = 5, min_best_conf: float = 0.85):
        self.max_rules = max_rules
        self.min_best_conf = min_best_conf

    def assess(self, rules: list) -> str:
        if not rules:
            return "greedy"
        if len(rules) > self.max_rules:
            return "short_k"
        best = max(r.confidence for r in rules)
        return "short_k" if best < self.min_best_conf else "full_k"


class RecoveryModeSelector:
    def __init__(self, calibration: int = 40):
        self.cal = calibration
        self.stats = {
            "truncate": {"time": 0.0, "tokens": 0, "rounds": 0},
            "seq_bonus": {"time": 0.0, "tokens": 0, "rounds": 0},
        }
        self.round = 0
        self.locked = None

    def get_mode(self) -> str:
        return self.locked if self.locked else ("truncate" if self.round % 2 == 0 else "seq_bonus")

    def record(self, mode: str, accepted_tokens: int, elapsed: float):
        if self.locked:
            return
        s = self.stats[mode]
        s["time"] += elapsed
        s["tokens"] += max(accepted_tokens, 1)
        s["rounds"] += 1
        self.round += 1
        if self.round >= self.cal and not self.locked:
            scores = {m: s["time"] / s["tokens"] for m, s in self.stats.items() if s["tokens"] > 0}
            if scores:
                self.locked = min(scores, key=scores.get)
                print(
                    f"[Recovery] locked={self.locked}  scores: "
                    f"{ {k: f'{v*1000:.2f}ms/tok' for k, v in scores.items()} }"
                )


class LastTargetTopKFilter:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._top_ids: set[int] = set()
        self._ready = False

    def update(self, last_logits_row: np.ndarray):
        top = np.argpartition(-last_logits_row, self.top_k)[: self.top_k]
        self._top_ids = set(top.tolist())
        self._ready = True

    def first_token_ok(self, token_id: int) -> bool:
        if not self._ready:
            return True
        return token_id in self._top_ids


class LiveNgramMiner:
    """Tracks only n-gram continuations that were VERIFIED accepted by target model."""

    def __init__(self, max_n: int = 4, window: int = 128):
        self.max_n = max_n
        self.window = window
        self._history: list[int] = []
        self._next_counts: dict[tuple[int, ...], dict[int, int]] = {}

    def feed_accepted(self, tokens: list[int]):
        self._history.extend(tokens)
        if len(self._history) > self.window:
            self._history = self._history[-self.window :]
        for i in range(len(tokens)):
            abs_i = len(self._history) - len(tokens) + i
            if abs_i == 0:
                continue
            for n in range(1, self.max_n + 1):
                if abs_i < n:
                    continue
                gram = tuple(self._history[abs_i - n : abs_i])
                nxt = self._history[abs_i]
                self._next_counts.setdefault(gram, {})
                self._next_counts[gram][nxt] = self._next_counts[gram].get(nxt, 0) + 1

    def propose(self, context: list[int], max_k: int) -> list[int] | None:
        if not self._history or max_k <= 0:
            return None
        result: list[int] = []
        tmp = list(context)
        for _ in range(min(max_k, 2)):
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
            if best_cnt / total < 0.80:
                break
            result.append(best_tok)
            tmp.append(best_tok)
        return result if result else None


class GreedyCorrectionSampler:
    @staticmethod
    def batch_verify(target_logits: np.ndarray, draft_ids: list[int], ctx_len: int):
        """Return (accepted_count, reject_pos, correction_token)."""
        num_draft = len(draft_ids)
        preds = np.argmax(target_logits[ctx_len : ctx_len + num_draft], axis=1)
        for i, (pred, draft) in enumerate(zip(preds, draft_ids)):
            if int(pred) != draft:
                return i, i, int(pred)
        bonus = int(np.argmax(target_logits[ctx_len + num_draft]))
        return num_draft, -1, bonus


class SpeedTraceCollector:
    """Writes one JSON line per prompt to a JSONL file."""

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
                    ctx = tuple(seq[i - n : i])
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
        tail = tokens[-self.max_ctx :] if len(tokens) >= self.max_ctx else tokens
        tail_len = len(tail)
        for n in range(upto, 0, -1):
            bucket = self.rules_by_len.get(n)
            if bucket is None:
                continue
            ctx = tuple(tail[tail_len - n :]) if tail_len >= n else tuple(tokens[-n:])
            rule = bucket.get(ctx)
            if rule is None:
                continue
            if rule.tier == "det_ctx6" and rule.support < 10:
                continue
            if banned is None or self.rule_key(rule) not in banned:
                return rule
        return None

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
            tier_cap = self.TIER_K_CAPS.get(rule.tier, max_k)
            if step >= tier_cap:
                break
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
    _RE_SUPER_INIT = re.compile(r"\bsuper\(\).__init__\($")
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
    _RE_SELF_NONE_ATTR = re.compile(
        r"\bself\.(data|key|value|val|left|right|next|prev|head|tail|root|node|parent|size|count|length)\s*=\s*$"
    )
    _RE_SELF_LIST_ATTR = re.compile(
        r"\bself\.(items|heap|stack|queue|arr|data|memo|cache|freq|dp|res|output|path)\s*=\s*$"
    )
    _RE_IF_NOT_SELF = re.compile(
        r"\bif\s+not\s+self\.(head|root|left|right|data|items|heap|stack|queue|node|val|value|key|next|prev|tail)\s*:\s*$"
    )
    _RE_NOARG_METHOD = re.compile(
        r"\.(items|keys|values|reverse|clear|copy|lower|upper|isdigit|isalpha|isalnum|isspace|isupper|islower|title|capitalize|strip|lstrip|rstrip|sort|pop|append|extend|count|index|find|replace|split|join|encode|decode|startswith|endswith|format|zfill|center|ljust|rjust)\($"
    )
    _RE_RETURN_TYPE = re.compile(
        r"\)\s*->\s*(?:None|bool|int|str|float|list|dict|tuple|set|bytes|Any|List|Dict|Tuple|Set|Optional|Union|Callable|Iterable|Iterator|Generator)$"
    )
    _RE_SLICE_COLON = re.compile(r"\[::-\s*$")
    _RE_SLICE_ONE = re.compile(r"\[::-1$")
    _RE_CLASS_DEF_INIT = re.compile(r"class\s+\w+\s*:\n\s+def\s+$")
    _RE_FUNC_CLOSE = re.compile(r"^\s*def\s+\w+\([A-Za-z_]\w*\)$")
    _RE_ELSE_TRY_FINALLY = re.compile(r"^(else|try|finally)$")
    _RE_RETURN_TERMINAL = re.compile(r"^\s*return\s+(True|False|None)$")
    _RE_INCREMENT_SPACE = re.compile(r"^\s*[ijn]\s*\+=$")
    _RE_INCREMENT_END = re.compile(r"^\s*[ijn]\s*\+=\s*1$")
    _RE_WHILE_COND = re.compile(r"^\s*while\s+[A-Za-z_]\w*$")

    def __init__(self, tokenizer: Tokenizer, mode: str = "cluster"):
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_ctx = 32
        self.prompt_hint: str = ""
        self.token_ids = self._init_token_ids()
        self.space_tokens: dict[int, int] = {}
        for n in range(1, 33):
            ids = tokenizer.encode(" " * n)
            if len(ids) == 1:
                self.space_tokens[n] = ids[0]

    def _single(self, text: str) -> int | None:
        ids = self.tokenizer.encode(text)
        return ids[0] if len(ids) == 1 else None

    def _init_token_ids(self) -> dict[str, int | None]:
        pieces = [
            " ", " in", " range", "(", ")", "\n", ":\n", "class", "def", " def",
            "init", "__(", "self", "(self", " len", " ==", "1", "2", " 1", " =",
            " None", " True", " False", " Exception", " as", " e", ".__init__(",
            ' "__main__"', "0", " [", "]", "]\n", "len(", "return", "not", "is",
            ",", " other", "):\n", "):", "[::-", " []\n\n",
        ]
        return {p: self._single(p) for p in pieces}

    def _text_tail(self, tokens: list[int], n: int = 64) -> str:
        return "".join(self.tokenizer.piece(t) for t in tokens[-n:])

    @staticmethod
    def _previous_line_indent(text: str) -> int:
        lines = text.split("\n")
        if len(lines) < 2:
            return 0
        prev = lines[-2]
        return len(prev) - len(prev.lstrip(" "))

    def _space_token_for_next_indent(self, text: str) -> int | None:
        desired = self._previous_line_indent(text) + 4
        return self.space_tokens.get(desired - 1)

    @staticmethod
    def _last_nonblank_indent(text: str) -> int:
        for line in reversed(text.split("\n")):
            if line.strip():
                return len(line) - len(line.lstrip(" "))
        return 0

    def _body_indent_after_colon(self, text: str) -> int | None:
        if not text.endswith("\n"):
            return None
        lines = text.rstrip("\n").split("\n")
        if not lines:
            return None
        if lines[-1].strip():
            return None
        for line in reversed(lines[:-1]):
            if line.strip():
                if line.rstrip().endswith(":"):
                    indent = len(line) - len(line.lstrip(" "))
                    return self.space_tokens.get(indent + 3)
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

        class_method_indent = self._class_next_method_indent(text)
        if class_method_indent is not None:
            token = self.space_tokens.get(class_method_indent)
            tier = "syntax_class_method_indent"
            conf = 0.99

        if token is None and (text.endswith("```python\n") or text.endswith("``` python\n")):
            if self.prompt_hint == "class":
                choose("class", 0.97, "syntax_code_fence_class")
            elif self.prompt_hint == "def":
                choose("def", 0.97, "syntax_code_fence_def")
            elif re.search(r"(linked|tree|heap|stack|queue|bst|node|graph|sort|search)", text, re.I):
                choose("class", 0.97, "syntax_code_fence_class")
            else:
                choose("def", 0.97, "syntax_code_fence_def")

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
            elif self._RE_SELF_LIST_ATTR.search(line):
                choose(" []\n\n", 0.97, "syntax_self_list_init")
            elif self._RE_SLICE_COLON.search(line):
                choose("1", 1.00, "syntax_slice_one")
            elif self._RE_SLICE_ONE.search(line):
                choose("]\n", 0.98, "syntax_slice_close")
            elif text.endswith(":\n") or text.endswith("):\n") or text.endswith("]:\n"):
                token = self._space_token_for_next_indent(text)
                tier = "syntax_indent"
                conf = 0.98

        if token is None:
            return None
        ctx = tuple(tokens[-min(self.max_ctx, len(tokens)) :])
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
    if not draft and live_ngram is not None:
        ngram_draft = live_ngram.propose(tokens, max_k)
        if ngram_draft:
            draft = ngram_draft
            used = [Rule(tuple(tokens[-4:]), t, 0.85, 1, 1, "live_ngram") for t in ngram_draft]
    return draft, used


TRACE_FIELDS = [
    "prompt", "kind", "pos", "pending", "draft", "accepted_batch",
    "accepted_draft", "bonus", "next_model_id", "tier", "rule_chain",
    "rule_conf", "rule_support", "ctx_tail_ids", "ctx_tail_text",
    "draft_ids", "draft_text", "model_draft_ids", "model_draft_text",
    "next_model_text", "mismatch_at", "why", "rule_strikes", "draft_mistakes",
    "pattern_s", "decode_s", "verify_s",
]


def _ids_csv(ids: list[int] | tuple[int, ...] | int | str) -> str:
    if isinstance(ids, (list, tuple)):
        return " ".join(str(int(x)) for x in ids)
    return str(ids)


def _trace_text(tokenizer: Tokenizer, ids: list[int] | tuple[int, ...]) -> str:
    if not ids:
        return ""
    return tokenizer.decode([int(x) for x in ids]).replace("\n", "\\n")


def enrich_trace_records(tokenizer: Tokenizer, records: list[dict]) -> list[dict]:
    enriched = []
    for r in records:
        row = dict(r)
        ctx_ids = list(row.get("ctx_tail_ids", []))
        draft_ids = list(row.get("draft_ids", []))
        model_ids = list(row.get("model_draft_ids", []))
        next_id = row.get("next_model_id", row.get("bonus", ""))
        row["ctx_tail_text"] = _trace_text(tokenizer, ctx_ids)
        row["draft_text"] = _trace_text(tokenizer, draft_ids)
        row["model_draft_text"] = _trace_text(tokenizer, model_ids)
        row["next_model_text"] = _trace_text(tokenizer, [next_id]) if next_id != "" else ""
        row["ctx_tail_ids"] = _ids_csv(ctx_ids)
        row["draft_ids"] = _ids_csv(draft_ids)
        row["model_draft_ids"] = _ids_csv(model_ids)
        enriched.append(row)
    return enriched


def write_trace_csv(path: Path, prompt: str, records: list[dict], tokenizer: Tokenizer | None = None) -> None:
    if not records:
        return
    out_records = enrich_trace_records(tokenizer, records) if tokenizer is not None else records
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in out_records:
            writer.writerow({"prompt": prompt, **r})
