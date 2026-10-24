from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable


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


class PatternMiner:
    def __init__(
        self,
        max_ctx: int = 8,
        min_support: int = 2,
        min_conf: float = 0.78,
        det_conf: float = 0.96,
        min_rule_ctx: int = 2,
    ):
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
        return self.set_rules(rules)

    def set_rules(self, rules: dict[tuple[int, ...], Rule]) -> "PatternMiner":
        self.rules_by_ctx = rules
        self.rule_count_by_tier = Counter()
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
        draft.append(rule.token)
        used.append(rule)
        tmp.append(rule.token)
        if rule.tier in {"syntax_floor_div_space", "syntax_floor_div_two", "syntax_pluseq_one"}:
            break
        if syntax is not None and syntax.mode == "basic" and rule.tier.startswith("syntax"):
            break
    return draft, used
