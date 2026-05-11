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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
from llama_cpp import Llama, LlamaCache

try:
    from structspec.ast_proposer import Confidence, PythonAstProposer
    AST_PROPOSER_AVAILABLE = True
except ImportError:
    Confidence = None  # type: ignore[assignment]
    PythonAstProposer = None  # type: ignore[assignment]
    AST_PROPOSER_AVAILABLE = False

try:
    from .viz import RichVisualizer, TokenEvent, RICH_AVAILABLE
except ImportError:
    RICH_AVAILABLE = False
    RichVisualizer = None  # type: ignore[misc,assignment]
    TokenEvent = None  # type: ignore[misc,assignment]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# No hardcoded defaults — users must supply valid paths via CLI arguments.
# This keeps the tool portable across operating systems and environments.


PROMPTS = [
    "implement linked list in python only code no comments",
    "implement fibonacci in python, no comments, only code:",
    "implement BST in python, only code, no comments",
    "reverse a list in python, only code, no comments",
    "reverse linked list in python, no comments, only code",
    "write function in python to check given number prime or not only code no comments",
    "implement merge sort in python only code no comments",
    "implement quick sort in python only code no comments",
    "write function to check given string palindrome or not in python only code no comments",
    "implement deletion at end in linked list in python only code no comments",
    "implement insertion at beginning in linked list in python only code no comments",
    "implement deletion of node at beginning in linked list in python only code no comments",
    "implement min heap in python only code no comments",
    "implement max heap in python only code no comments",
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
        min_support: int = 1,          # relaxed from 2 → 1
        min_conf: float = 0.85,        # relaxed from 0.96 → 0.85
        det_conf: float = 0.96,
        min_rule_ctx: int = 4,         # raised from 2 → 4 (P3)
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

    def find_rule(self, tokens: list[int], banned: set[tuple[tuple[int, ...], int, str]] | dict | None = None) -> Rule | None:
        upto = min(self.max_ctx, len(tokens))
        for n in range(upto, 0, -1):
            rule = self.rules_by_len.get(n, {}).get(tuple(tokens[-n:]))
            if rule is not None and rule.tier == "det_ctx6" and rule.support < 10:
                continue
            if rule is not None and (banned is None or self.rule_key(rule) not in banned):
                return rule
        return None

    def propose(
        self,
        tokens: list[int],
        max_k: int,
        banned: set[tuple[tuple[int, ...], int, str]] | None = None,
    ) -> tuple[list[int], list[Rule]]:
        draft: list[int] = []
        used: list[Rule] = []
        tmp = list(tokens)
        for step in range(max_k):
            rule = self.find_rule(tmp, banned=banned)
            if rule is None:
                break
            # Long drafts must be deterministic-ish. Low-confidence rules are
            # allowed as the first guess only; otherwise they waste target eval.
            if step > 0 and (rule.confidence < self.det_conf or rule.support < 3):
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


class IndentStack:
    def __init__(self):
        self._stack: list[int] = [0]

    def reset(self) -> None:
        self._stack[:] = [0]

    def current(self) -> int:
        return self._stack[-1]

    def push(self, level: int) -> None:
        self._stack.append(level)

    def pop(self) -> int:
        if len(self._stack) > 1:
            return self._stack.pop()
        return self._stack[-1]

    def dedent_to(self, target: int) -> None:
        while len(self._stack) > 1 and self._stack[-1] > target:
            self._stack.pop()


class PythonSyntaxProposer:
    """Tiny Qwen-token syntax backoff for common Python code moves."""

    def __init__(self, model: "FastGreedyLlama", mode: str = "cluster"):
        self.model = model
        self.mode = mode
        self.max_ctx = 32
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
        }
        self.space_tokens: dict[int, int] = {}
        for n in range(1, 33):
            ids = model.tokenize(" " * n)
            if len(ids) == 1:
                self.space_tokens[n] = ids[0]
        self._running_text = ""
        self._indent_stack = IndentStack()

    def reset_cache(self) -> None:
        self._running_text = ""
        self._indent_stack.reset()

    def feed_token(self, piece: str) -> None:
        self._running_text += piece

    def _single(self, text: str) -> int | None:
        ids = self.model.tokenize(text)
        return ids[0] if len(ids) == 1 else None

    def _sync_indent_stack(self, text: str) -> None:
        lines = text.split("\n")
        if len(lines) < 2:
            return
        prev = lines[-2].rstrip()
        prev_indent = len(lines[-2]) - len(lines[-2].lstrip(" "))
        curr_indent = len(lines[-1]) - len(lines[-1].lstrip(" "))

        if prev.endswith(":") and lines[-2].strip():
            expected = prev_indent + 4
            if expected != self._indent_stack.current():
                self._indent_stack.push(expected)
        elif prev_indent > 0 and prev.strip().startswith(
            ("return", "raise", "break", "continue", "pass")
        ):
            self._indent_stack.dedent_to(max(0, prev_indent - 4))

    def _text_tail(self, tokens: list[int], n: int = 64) -> str:
        if self._running_text:
            return self._running_text[-min(n, len(self._running_text)):]
        return "".join(self.model.token_piece(t) for t in tokens[-n:])

    @staticmethod
    def _previous_line_indent(text: str) -> int:
        lines = text.split("\n")
        if len(lines) < 2:
            return 0
        prev = lines[-2]
        return len(prev) - len(prev.lstrip(" "))

    def _space_token_for_next_indent(self, text: str) -> int | None:
        # Qwen commonly encodes one indentation space on the following word
        # token, so the pure-space token is desired_indent - 1.
        desired = self._previous_line_indent(text) + 3
        return self.space_tokens.get(desired)

    def _space_token_for_visual_indent(self, indent: int) -> int | None:
        if indent <= 0:
            return None
        return self.space_tokens.get(max(1, indent - 1))

    def _last_nonblank_indent(self, text: str) -> int:
        for line in reversed(text.split("\n")):
            if line.strip():
                return len(line) - len(line.lstrip(" "))
        return 0

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

    def _indent_after_plain_newline(self, text: str) -> tuple[int | None, float, str]:
        if not text.endswith("\n") or text.endswith(("\n\n", ":\n", "):\n", "]:\n")):
            return None, 0.0, ""
        self._sync_indent_stack(text)
        lines = text.split("\n")
        if len(lines) < 2 or lines[-1] != "":
            return None, 0.0, ""
        prev = lines[-2]
        stripped = prev.strip()
        if not stripped:
            return None, 0.0, ""
        indent = len(prev) - len(prev.lstrip(" "))

        # After a terminal control statement, use the indent stack to handle
        # multi-level dedent (e.g. return inside nested if/for).
        if stripped.startswith(("return", "raise", "break", "continue", "pass")) and indent > 0:
            dedent_to = self._indent_stack.current()
            return self._space_token_for_visual_indent(dedent_to), 0.94, "syntax_dedent_after_terminal"

        # In constructors Qwen often emits several self-field assignments in a
        # row. The next token is just the same indentation before " self".
        if indent >= 8 and re.match(r"self\.[A-Za-z_][A-Za-z0-9_]*\s*=", stripped):
            return self._space_token_for_visual_indent(indent), 0.93, "syntax_same_indent_self_assign"

        # In loop bodies, repeated simple assignments commonly stay at the same
        # indentation. Keep this below syntax/mined deterministic rules.
        if indent >= 8 and re.match(r"[A-Za-z_][A-Za-z0-9_]*\s*(=|\+=|-=)", stripped):
            return self._space_token_for_visual_indent(indent), 0.88, "syntax_same_indent_simple_assign"

        return None, 0.0, ""

    def find_rule(
        self,
        tokens: list[int],
        banned: set[tuple[tuple[int, ...], int, str]] | dict | None = None,
    ) -> Rule | None:
        if not tokens:
            return None
        text = self._text_tail(tokens)
        self._sync_indent_stack(text)
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

        # P3: Guard against false positives when exiting a class
        # Only apply class method indent detection if we're still inside a class (indent == 0 on current line)
        if class_method_indent is not None:
            # Additional check: ensure the previous non-blank line had class keyword
            lines = text.split("\n")
            # Find the last non-blank line before current
            for prev_line in reversed(lines[:-1]):
                if prev_line.strip():
                    # If that line is at indent 0 and contains 'class', we're inside a class
                    if len(prev_line) - len(prev_line.lstrip(" ")) == 0 and prev_line.strip().startswith("class "):
                        token = self.space_tokens.get(class_method_indent)
                        tier = "syntax_class_method_indent"
                        conf = 0.99
                    break
            if token is None:
                # Not actually inside a class, skip this rule
                pass
            else:
                # token already set, will return below
                pass
        elif text.endswith("```python\n") or text.endswith("``` python\n"):
            if re.search(r"(linked|tree|heap|stack|queue|bst|node)", text, re.I):
                choose("class", 0.90, "syntax_code_fence_class")
            else:
                choose("def", 0.90, "syntax_code_fence_def")
        elif text.endswith("\n\n   "):
            choose(" def", 1.00, "syntax_class_next_method")
        elif text.endswith((":\n", "):\n", "]:\n")):
            token = self._space_token_for_next_indent(text)
            tier = "syntax_indent"
            conf = 0.98
        else:
            token, conf, tier = self._indent_after_plain_newline(text)

        if token is None and line.rstrip().endswith("//"):
            choose(" ", 1.00, "syntax_floor_div_space")
        elif token is None and re.search(r"//\s$", line):
            choose("2", 1.00, "syntax_floor_div_two")
        elif token is None and re.match(
            r"^\s*(i|j|k|n|idx|index|count|cnt|left|right)\s*\+=$",
            line.rstrip(),
        ):
            choose(" ", 1.00, "syntax_pluseq_space")
        elif token is None and re.match(
            r"^\s*(i|j|k|n|idx|index|count|cnt|left|right)\s*\+=\s$",
            line,
        ):
            choose("1", 1.00, "syntax_pluseq_one")
        elif token is None and re.match(r"^\s*for\s+[A-Za-z_][A-Za-z0-9_]*$", line):
            choose(" in", 0.99, "syntax_for_in")
        elif token is None and re.match(
            r"^\s*for\s+(i|j|k|idx|index|n|num|count)\s+in$",
            line,
        ):
            choose(" range", 0.97, "syntax_for_range")
        elif token is None and re.match(r"^\s*for\b.*\brange$", line):
            choose("(", 1.00, "syntax_range_paren")
        elif token is None and re.search(r"\bwhile\s+True$", line):
            choose(":\n", 1.00, "syntax_while_true")
        elif token is None and re.search(r"\bif\s+__name__$", line):
            choose(" ==", 1.00, "syntax_name_eq")
        elif token is None and re.search(r"\bif\s+__name__\s+==$", line):
            choose(' "__main__"', 1.00, "syntax_name_main")
        elif token is None and re.search(r'\bif\s+__name__\s+==\s+"__main__"$', line):
            choose(":\n", 1.00, "syntax_main_colon")
        elif token is None and stripped == "try":
            choose(":\n", 1.00, "syntax_try_colon")
        elif token is None and stripped == "except":
            choose(" Exception", 0.88, "syntax_except_exception")
        elif token is None and stripped == "except Exception":
            choose(" as", 0.90, "syntax_except_as")
        elif token is None and stripped == "except Exception as":
            choose(" e", 0.98, "syntax_except_e")
        elif token is None and stripped == "except Exception as e":
            choose(":\n", 1.00, "syntax_except_colon")
        elif token is None and re.search(r"\bsuper\(\)$", line):
            choose(".__init__(", 0.98, "syntax_super_init")
        elif token is None and re.search(r"\bsuper\(\)\.__init__\($", line):
            choose(")", 0.90, "syntax_super_close")
        elif token is None and re.search(r"\bdef\s+__$", line):
            choose("init", 1.00, "syntax_dunder_init")
        elif token is None and re.search(r"\bdef\s+__init$", line):
            choose("__(", 1.00, "syntax_dunder_paren")
        elif token is None and re.search(r"\bdef\s+__init__\($", line):
            choose("self", 1.00, "syntax_dunder_self")
        elif token is None and re.search(r"\breturn\s+len$", line):
            choose("(self", 1.00, "syntax_return_len_self")
        elif token is None and (
            re.match(r"^(def|class)\b", stripped)
            and stripped.endswith(")")
            and ":" not in stripped
            and not stripped.endswith(":")
        ):
            choose(":\n", 0.99, "syntax_def_colon")
        elif token is None and stripped in {"else", "try", "finally"}:
            choose(":\n", 1.00, "syntax_block_colon")
        elif token is None and re.match(r"^\s*return\s+(True|False|None)$", line):
            choose("\n", 1.00, "syntax_return_terminal")
        elif token is None and stripped in {"break", "continue", "pass"}:
            choose("\n", 1.00, "syntax_statement_end")
        elif token is None and re.match(r"^\s*[ijn]\s*\+=$", line):
            choose(" ", 1.00, "syntax_increment_space")
        elif token is None and re.match(r"^\s*[ijn]\s*\+=\s*1$", line):
            choose("\n", 1.00, "syntax_increment_end")
        elif token is None and re.search(r"\(n\s*-\s*1\)\s*\+\s*f\(n\s*-$", line):
            choose("2", 1.00, "syntax_fib_n_minus_two")
        elif token is None and re.search(r"(\[.*-|\brange\(.*-|\bn\s*-)$", line):
            choose("1", 0.93, "syntax_minus_one")
        elif token is None and re.match(r"^.*\bif\b.*\bis\s+(not\s+)?None$", line):
            choose(":\n", 0.97, "syntax_none_guard_colon")
        elif token is None and re.match(r"^.*\)\s+as\s+[A-Za-z_][A-Za-z0-9_]*$", line):
            choose(":\n", 0.97, "syntax_with_as_colon")

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
    banned: set[tuple[tuple[int, ...], int, str]] | dict | None = None,
    adaptive_k: bool = False,
    ast_proposer: "PythonAstProposer | None" = None,
    tokenize_fn: "Callable[[str], list[int]] | None" = None,
) -> tuple[list[int], list[Rule]]:
    draft: list[int] = []
    used: list[Rule] = []
    tmp = list(tokens)
    ctx_tail = tuple(tmp[-8:])
    chain_limit = max_k

    # Try AST proposer first for HARD/MEDIUM grammar-guaranteed tokens
    if ast_proposer is not None and tokenize_fn is not None:
        ast_proposal = ast_proposer.propose(max_k=max_k)
        for pred in ast_proposal.tokens:
            if len(draft) >= max_k:
                break
            tok_ids = tokenize_fn(pred.text)
            if not tok_ids:
                continue
            for tid in tok_ids:
                if len(draft) >= max_k:
                    break
                draft.append(tid)
                used.append(Rule(ctx_tail, tid, 0.99, 999, 1000, "ast_" + pred.reason))
            # MEDIUM predictions stop the chain after one token
            if pred.confidence == Confidence.MEDIUM:
                break
        if draft:
            return draft, used

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
        if adaptive_k:
            rule_limit = adaptive_rule_k(rule, max_k)
            if rule_limit <= 0:
                break
            chain_limit = min(chain_limit, rule_limit)
            if step >= chain_limit:
                break
        if step > 0:
            is_high_trust = (
                rule.tier.startswith("syntax_")
                or (rule.tier in {"det_ctx8", "det_ctx7", "det_ctx6"} and rule.confidence >= 0.97)
            )
            if not is_high_trust and (rule.confidence < miner.det_conf or rule.support < 3):
                break
        draft.append(rule.token)
        used.append(rule)
        tmp.append(rule.token)
        if adaptive_k and len(draft) >= chain_limit:
            break
        if rule.tier in {"syntax_floor_div_two", "syntax_pluseq_one"}:
            break
        if syntax is not None and syntax.mode == "basic" and rule.tier.startswith("syntax"):
            break
    return draft, used


def adaptive_rule_k(rule: Rule, max_k: int) -> int:
    """Cap draft chain length by the weakest rule seen so far.

    The caps are intentionally conservative for HumanEval-style prompts, where
    the current DSA corpus has many accidental shallow-context collisions.
    """
    tier = rule.tier
    if tier in {"syntax_range_paren", "syntax_class_next_method"}:
        return 0
    if tier in {"syntax_pluseq_space", "syntax_for_range", "syntax_same_indent_simple_assign"}:
        return min(max_k, 1)
    if tier in {
        "syntax_indent",
        "syntax_name_eq",
        "syntax_name_main",
        "syntax_main_colon",
        "syntax_block_colon",
        "syntax_none_guard_colon",
        "syntax_def_colon",
        "syntax_dunder_init",
        "syntax_dunder_paren",
        "syntax_dunder_self",
    }:
        return min(max_k, 8)
    if tier in {"syntax_return_terminal", "syntax_minus_one", "syntax_for_in"}:
        return min(max_k, 2)
    if tier.startswith("syntax_"):
        return min(max_k, 4)

    if rule.support < 3:
        return min(max_k, 1)
    if tier == "det_ctx8":
        return min(max_k, 6)
    if tier == "strong_ctx8":
        return min(max_k, 4)
    if tier in {"det_ctx7", "strong_ctx7"}:
        return min(max_k, 3)
    if tier in {"det_ctx6", "strong_ctx6"}:
        return min(max_k, 2)
    if tier in {"det_ctx5", "strong_ctx5", "det_ctx4", "strong_ctx4"}:
        return min(max_k, 1)
    return min(max_k, 2)


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
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        flash_attn: bool = False,
    ):
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            logits_all=False,
            flash_attn=flash_attn,
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
        if n_tokens == self.llm.n_tokens:
            return
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
    k: int,
    reject_mode: str = "truncate",
    strike_limit: int = 3,
    draft_model: FastGreedyLlama | None = None,
    draft_mistakes: Counter[tuple[tuple[int, ...], int]] | None = None,
    draft_mistake_limit: int = 1,
    live_viz: bool = False,
    rich_viz: bool = False,
    visualizer: Optional[RichVisualizer] = None,
    adaptive_k: bool = False,
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
    accept_hist = Counter()
    banned_rules: dict[tuple[tuple[int, ...], int, str], int] = {}
    rule_strikes: Counter[tuple[tuple[int, ...], int, str]] = Counter()
    draft_mistakes = draft_mistakes if draft_mistakes is not None else Counter()
    BAN_DECAY = 50  # unban rules after this many generated tokens
    if syntax is not None:
        syntax.reset_cache()
        syntax.feed_token(prompt)

    # Initialize AST proposer for grammar-based predictions
    ast_proposer = None
    if AST_PROPOSER_AVAILABLE and PythonAstProposer is not None:
        ast_proposer = PythonAstProposer(max_chain=k)
        ast_proposer.observe_prompt(prompt)
        ast_proposer.commit(prompt)
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
                f"\n[warn] pending={pending} > 1 (position {len(gen)}), "
                f"falling back to greedy for this step",
                file=sys.stderr,
                flush=True,
            )
            gen = gen[:kv_len]
            pending = []
            model.truncate_kv(kv_len)
            if gen:
                t_recover = now()
                recover_logits = model.decode_logits(gen[-1:], logits_all=False)[0]
                times["decode"] += now() - t_recover
                passes += 1
                prev_pred = model.argmax(recover_logits)

        # If there is no pending token and no useful draft, emit the model's
        # already-known greedy token as pending. Next loop can draft after it.
        max_draft = max(0, min(k, remaining - 1))
        ctx_tail_ids = gen[max(0, len(gen) - 12):]
        pos = len(gen) - len(prompt_ids)

        # Decay banned rules that are older than BAN_DECAY tokens
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
            adaptive_k=adaptive_k,
            ast_proposer=ast_proposer,
            tokenize_fn=model.tokenize,
        )
        if not draft and draft_model is not None and max_draft > 0:
            draft, rules = propose_from_draft_model(
                draft_model, gen, max_draft, draft_mistakes, draft_mistake_limit
            )
        pattern_dt = now() - t0
        times["pattern"] += pattern_dt

        if not pending and not draft:
            gen.append(prev_pred)
            greedy_emitted += 1
            if syntax is not None:
                syntax.feed_token(model.token_piece(prev_pred))
            if ast_proposer is not None:
                ast_proposer.commit(model.token_piece(prev_pred))
            if prev_pred == model.eos or len(gen) >= target:
                break
            continue

        batch = pending + draft
        if not batch:
            gen.append(prev_pred)
            greedy_emitted += 1
            if syntax is not None:
                syntax.feed_token(model.token_piece(prev_pred))
            if ast_proposer is not None:
                ast_proposer.commit(model.token_piece(prev_pred))
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

        if rejected and reject_mode == "rebuild":
            accepted_batch = accepted_pending
            accepted_draft = 0
            clean_prefix = gen[:old_kv] + batch[:accepted_batch]
            model.reset()
            if syntax is not None:
                syntax.reset_cache()
                syntax.feed_token(prompt)
                for tok in clean_prefix[len(prompt_ids):]:
                    syntax.feed_token(model.token_piece(tok))
            if ast_proposer is not None:
                ast_proposer = PythonAstProposer(max_chain=k)
                ast_proposer.observe_prompt(prompt)
                ast_proposer.commit(prompt)
                for tok in clean_prefix[len(prompt_ids):]:
                    ast_proposer.commit(model.token_piece(tok))
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
            else:
                model.truncate_kv(old_kv)
                bonus = prev_pred
        elif rejected and reject_mode == "truncate":
            pass

        proposed += len(draft)
        accepted += accepted_draft
        draft_starts += int(bool(draft))
        accept_hist[accepted_draft] += int(bool(draft))
        reject_count += int(rejected)

        gen = gen[:old_kv] + batch[:accepted_batch] + [bonus]
        kv_len = old_kv + accepted_batch
        model.truncate_kv(kv_len)
        prev_pred = bonus

        if syntax is not None:
            for tok in batch[:accepted_batch]:
                syntax.feed_token(model.token_piece(tok))
            syntax.feed_token(model.token_piece(bonus))
        if ast_proposer is not None:
            for tok in batch[:accepted_batch]:
                ast_proposer.commit(model.token_piece(tok))
            ast_proposer.commit(model.token_piece(bonus))

        tokens_generated = len(gen) - len(prompt_ids)

        if rich_viz and visualizer and RICH_AVAILABLE:
            events: list[TokenEvent] = []
            for j in range(accepted_pending):
                tok = batch[j]
                events.append(TokenEvent(model.token_piece(tok), "verified"))
            for j in range(accepted_draft):
                tok = draft[j]
                tier = rules[j].tier if j < len(rules) else ""
                events.append(TokenEvent(model.token_piece(tok), "accepted", tier))
            if rejected and accepted_draft < len(draft):
                tok = draft[accepted_draft]
                tier = rules[accepted_draft].tier if accepted_draft < len(rules) else ""
                events.append(TokenEvent(model.token_piece(tok), "rejected", tier))
            events.append(TokenEvent(model.token_piece(bonus), "bonus"))
            visualizer.update(
                current_len=tokens_generated,
                passes=passes,
                accepted_draft=accepted,
                proposed=proposed,
                token_events=events,
            )
        elif live_viz and (draft or rejected):
            reduced = accepted_draft
            total_speed = tokens_generated / max(1, passes)
            print(
                f"\r[live] tok={tokens_generated:>4}/{max_tokens} "
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
            "passes_saved_est": accepted_draft,
            "pattern_s": pattern_dt,
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
    "passes_saved_est", "pattern_s", "decode_s", "verify_s",
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


def _check_file(path: str, label: str) -> Path:
    p = Path(path)
    if not p.exists():
        print(
            f"ERROR: {label} not found: {p}\n"
            f"       Please provide a valid path via the --{label.replace(' ', '-')} argument.\n"
            f"       Run with --help for usage details.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not p.is_file():
        print(f"ERROR: {label} is not a file: {p}", file=sys.stderr)
        raise SystemExit(2)
    return p


def _model_compatibility_warning(model_path: str) -> None:
    lower = model_path.lower()
    if "qwen" not in lower:
        print(
            "\nWARNING: The current pattern miner and syntax rules are optimized for Qwen models.\n"
            "         Using a different model may produce sub-optimal or incorrect results.\n"
            "         See README.md for guidance on adapting StructSpec to other models.\n",
            file=sys.stderr,
        )


def benchmark(args: argparse.Namespace) -> None:
    model_path = _check_file(args.model, "model")
    token_json_path = _check_file(args.token_json, "token-json")
    _model_compatibility_warning(str(model_path))

    corpus = QwenTokenCorpus(token_json_path)
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
    model = FastGreedyLlama(
        args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        flash_attn=args.flash_attn,
    )
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
        draft_model = FastGreedyLlama(
            args.draft_model,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            flash_attn=args.flash_attn,
        )
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

    if RICH_AVAILABLE:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel.fit(
            "[bold blue]StructSpec[/] — Zero-VRAM Structural Speculative Decoding",
            subtitle="Benchmark Suite",
            border_style="blue",
        ))

    print("\nCorrect pending-bonus speculative benchmark")
    print(f"  K={args.k}  max_tokens={args.tokens}  prompts={len(prompts)}")
    print("  NOTE: pass = actual target decode call. Bonus is NOT eval'd separately.\n")

    for idx, prompt in enumerate(prompts, 1):
        g_ids, g_text, gs = run_greedy(model, prompt, args.tokens)
        greedy_speed = args.tokens / max(1e-9, gs["time"]) if gs["time"] > 0 else None
        prompt_ids_for_mining = model.tokenize(prompt)
        if args.mine_greedy_before_spec:
            train_sequences.append(g_ids[len(prompt_ids_for_mining):])
            refit_miner(miner, train_sequences)

        visualizer = None
        if args.rich_viz:
            if RICH_AVAILABLE:
                visualizer = RichVisualizer(
                    max_tokens=args.tokens,
                    greedy_reference_speed=greedy_speed,
                )
                visualizer.__enter__()
            else:
                print("Warning: --rich-viz requires rich. Install with: pip install rich")

        try:
            s_ids, s_text, ss = run_speculative(
                model, miner, syntax, prompt, args.tokens, args.k,
                reject_mode=reject_mode,
                strike_limit=args.strike_limit,
                draft_model=draft_model,
                draft_mistakes=draft_mistakes,
                draft_mistake_limit=args.draft_mistake_limit,
                live_viz=args.live_viz,
                rich_viz=args.rich_viz,
                visualizer=visualizer,
                adaptive_k=args.adaptive_k,
            )
        finally:
            if visualizer:
                visualizer.__exit__(None, None, None)
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

    if RICH_AVAILABLE:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(
            title="Aggregate Results",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Greedy", justify="right")
        table.add_column("Speculative", justify="right")
        table.add_column("Speedup", justify="right", style="green")
        table.add_row(
            "Passes",
            str(pass_g),
            str(pass_s),
            f"{pass_g / max(1, pass_s):.2f}×",
        )
        table.add_row(
            "Wall Time",
            f"{time_g:.2f}s",
            f"{time_s:.2f}s",
            f"{time_g / max(1e-9, time_s):.2f}×",
        )
        table.add_row(
            "Eval Tokens",
            str(eval_g),
            str(eval_s),
            f"{eval_g / max(1, eval_s):.2f}×",
        )
        table.add_row(
            "Identical",
            f"{ok}/{len(prompts)}",
            "",
            "",
        )
        console.print(table)

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

    if args.json_output:
        summary = {
            "identical_outputs": ok,
            "total_prompts": len(prompts),
            "passes_greedy": pass_g,
            "passes_spec": pass_s,
            "pass_speedup": pass_g / max(1, pass_s),
            "eval_greedy": eval_g,
            "eval_spec": eval_s,
            "eval_ratio": eval_g / max(1, eval_s),
            "time_greedy": time_g,
            "time_spec": time_s,
            "time_speedup": time_g / max(1e-9, time_s),
            "draft_accepted": accepted_total if all_records else 0,
            "draft_proposed": proposed_total if all_records else 0,
            "draft_rejected": reject_total if all_records else 0,
            "no_rule_passes": no_rule_total if all_records else 0,
        }
        print(f"STRUCTSPEC_JSON:{json.dumps(summary)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to target model GGUF file.")
    ap.add_argument("--token-json", required=True, help="Path to token corpus JSON file.")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-ctx", type=int, default=8)
    ap.add_argument("--min-support", type=int, default=1, help="Minimum rule support (default: 1)")
    ap.add_argument("--min-conf", type=float, default=0.85, help="Minimum rule confidence (default: 0.85)")
    ap.add_argument("--det-conf", type=float, default=0.96)
    ap.add_argument("--min-rule-ctx", type=int, default=4, help="Minimum rule context length (default: 4)")
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--flash-attn", action="store_true",
                    help="Enable llama.cpp flash attention. Needed by some GGUF architectures for batched decode.")
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
    ap.add_argument("--rich-viz", action="store_true",
                    help="Enable Rich live terminal visualization with token colors and metrics.")
    ap.add_argument("--adaptive-k", action="store_true",
                    help="Cap draft chain length by rule trust to avoid verifying long weak drafts.")
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
    ap.add_argument("--trace-csv", default="qwen_spec_trace.csv",
                    help="Path for the output CSV trace file (default: qwen_spec_trace.csv in cwd).")
    ap.add_argument("--json-output", action="store_true",
                    help="Emit a final JSON summary line prefixed with 'STRUCTSPEC_JSON:' for programmatic consumption.")
    return ap.parse_args()


def main() -> None:
    benchmark(parse_args())


if __name__ == "__main__":
    main()
