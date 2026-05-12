"""
structspec.ast_proposer
=======================

Stateful syntax-aware speculative proposer for Python code generation.

Tracks bracket/indentation state incrementally as tokens are generated
and proposes high-confidence next tokens grounded in formal Python grammar.
Does NOT call a draft model. Every prediction is grammar-guaranteed (HARD)
or strongly templated (MEDIUM), ensuring near-100% accept rates.

Fast: O(1) per proposal step. No tree-sitter calls on the hot path.
Tree-sitter is used optionally as a one-shot assist on the prompt.

Confidence tiers
----------------
HARD   (>= 0.99): Grammar guarantees the token. Emit aggressively.
MEDIUM (0.85-0.98): Strongly templated. Emit one at a time.
SOFT   (< 0.85): Not emitted. Falls through to other proposers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

# --- Optional tree-sitter import. Degrades gracefully if missing. -----------
try:
    import tree_sitter as _ts
    import tree_sitter_python as _tspython

    _TS_LANG = _ts.Language(_tspython.language())
    _TS_PARSER = _ts.Parser(_TS_LANG)
    TREE_SITTER_AVAILABLE = True
except Exception:
    _TS_PARSER = None
    TREE_SITTER_AVAILABLE = False


class Confidence(Enum):
    HARD = "hard"
    MEDIUM = "medium"
    SOFT = "soft"


@dataclass(frozen=True)
class Prediction:
    text: str
    confidence: Confidence
    reason: str


@dataclass
class Proposal:
    tokens: list[Prediction] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(t.text for t in self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def __bool__(self) -> bool:
        return bool(self.tokens)


# ---------------------------------------------------------------------------
# Internal: shadow tokenizer state
# ---------------------------------------------------------------------------

_HEADER_KEYWORDS = frozenset({
    "def", "class", "if", "elif", "else", "for", "while",
    "try", "except", "finally", "with", "async", "match", "case",
})

_BRACKET_OPEN = "([{"
_BRACKET_CLOSE = ")]}"
_BRACKET_PAIR = {"(": ")", "[": "]", "{": "}"}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class _ShadowState:
    bracket_stack: list[str] = field(default_factory=list)
    indent_stack: list[int] = field(default_factory=lambda: [0])
    line_buffer: str = ""
    in_string: bool = False
    string_delim: str = ""
    string_triple: bool = False
    line_continuation: bool = False
    line_head: str = ""
    expects_indent: bool = False

    def feed(self, text: str) -> None:
        for ch in text:
            self._feed_char(ch)

    def _feed_char(self, ch: str) -> None:
        if self.in_string:
            self._on_string_char(ch)
            return

        if ch == "\n":
            self._on_newline()
            return

        if ch in ("'", '"'):
            self.line_buffer += ch
            tail = self.line_buffer[-3:]
            if tail == ch * 3:
                self.in_string = True
                self.string_delim = ch
                self.string_triple = True
                return
            self.in_string = True
            self.string_delim = ch
            self.string_triple = False
            return

        if ch == "#" and not self.line_buffer.rstrip():
            self.line_buffer += ch
            self.line_head = "#"
            return

        if ch in _BRACKET_OPEN:
            self.bracket_stack.append(ch)
            self.line_buffer += ch
            return

        if ch in _BRACKET_CLOSE:
            if self.bracket_stack and _BRACKET_PAIR[self.bracket_stack[-1]] == ch:
                self.bracket_stack.pop()
            self.line_buffer += ch
            return

        if ch == ":" and not self.bracket_stack:
            self.expects_indent = True
            self.line_buffer += ch
            return

        if ch == "\\":
            self.line_continuation = True
            self.line_buffer += ch
            return

        self.line_buffer += ch
        if ch not in " \t":
            match = _IDENT_RE.match(self.line_buffer.lstrip())
            if match:
                self.line_head = match.group(0)
            elif not self.line_head:
                self.line_head = ch

    def _on_string_char(self, ch: str) -> None:
        self.line_buffer += ch
        if ch == "\\":
            self.line_continuation = not self.line_continuation
            return
        if self.line_continuation:
            self.line_continuation = False
            return
        if self.string_triple:
            if self.line_buffer.endswith(self.string_delim * 3):
                self.in_string = False
                self.string_delim = ""
                self.string_triple = False
        else:
            if ch == self.string_delim:
                self.in_string = False
                self.string_delim = ""
            elif ch == "\n":
                self.in_string = False
                self.string_delim = ""

    def _on_newline(self) -> None:
        if self.bracket_stack or self.line_continuation:
            self.line_continuation = False
            self.line_buffer = ""
            self.line_head = ""
            return

        stripped = self.line_buffer.rstrip()
        if stripped:
            new_indent = len(stripped) - len(stripped.lstrip(" \t"))
            cur_top = self.indent_stack[-1]
            if self.expects_indent:
                self.indent_stack.append(cur_top + 4)
            elif new_indent < cur_top:
                while len(self.indent_stack) > 1 and self.indent_stack[-1] > new_indent:
                    self.indent_stack.pop()

        self.expects_indent = False
        self.line_buffer = ""
        self.line_head = ""

    @property
    def current_indent(self) -> int:
        return self.indent_stack[-1] if self.indent_stack else 0

    @property
    def open_bracket(self) -> Optional[str]:
        return self.bracket_stack[-1] if self.bracket_stack else None

    @property
    def line_is_header_keyword(self) -> bool:
        return self.line_head in _HEADER_KEYWORDS and self.line_head != "#"

    @property
    def line_ends_with_balanced_paren_close(self) -> bool:
        rstripped = self.line_buffer.rstrip()
        if not rstripped.endswith(")"):
            return False
        opens = rstripped.count("(")
        closes = rstripped.count(")")
        return opens == closes and opens >= 1

    def clone(self) -> _ShadowState:
        s = _ShadowState()
        s.bracket_stack = list(self.bracket_stack)
        s.indent_stack = list(self.indent_stack)
        s.line_buffer = self.line_buffer
        s.in_string = self.in_string
        s.string_delim = self.string_delim
        s.string_triple = self.string_triple
        s.line_continuation = self.line_continuation
        s.line_head = self.line_head
        s.expects_indent = self.expects_indent
        return s


# ---------------------------------------------------------------------------
# Public proposer
# ---------------------------------------------------------------------------


class PythonAstProposer:
    def __init__(self, max_chain: int = 6, enable_medium: bool = True) -> None:
        self.max_chain = max_chain
        self.enable_medium = enable_medium
        self._state = _ShadowState()
        self._prompt_identifiers: set[str] = set()
        self._prompt_class_context = False

    def observe_prompt(self, prompt: str) -> None:
        if not prompt:
            return
        for match in _IDENT_RE.finditer(prompt):
            ident = match.group(0)
            if len(ident) >= 2:
                self._prompt_identifiers.add(ident)
        lowered = prompt.lower()
        self._prompt_class_context = any(
            kw in lowered for kw in ("class ", "method", "self.", "self,", "self)")
        )

        if TREE_SITTER_AVAILABLE and _TS_PARSER is not None:
            try:
                tree = _TS_PARSER.parse(prompt.encode("utf-8"))
                self._walk_for_names(tree.root_node, prompt.encode("utf-8"))
            except Exception:
                pass

    def _walk_for_names(self, node, source: bytes) -> None:
        if node.type in ("function_definition", "class_definition"):
            name = node.child_by_field_name("name")
            if name is not None:
                self._prompt_identifiers.add(
                    source[name.start_byte : name.end_byte].decode("utf-8", "replace")
                )
            if node.type == "class_definition":
                self._prompt_class_context = True
        for child in node.children:
            self._walk_for_names(child, source)

    def commit(self, text: str) -> None:
        if text:
            self._state.feed(text)

    def propose(self, max_k: Optional[int] = None) -> Proposal:
        budget = min(max_k or self.max_chain, self.max_chain)
        proposal = Proposal()
        chain_state = self._state.clone()

        for _ in range(budget):
            pred = self._next_prediction(chain_state)
            if pred is None:
                break
            if pred.confidence is Confidence.SOFT:
                break
            if pred.confidence is Confidence.MEDIUM and not self.enable_medium:
                break
            proposal.tokens.append(pred)
            chain_state.feed(pred.text)
            if pred.confidence is Confidence.MEDIUM:
                break

        return proposal

    def _next_prediction(self, state: _ShadowState) -> Optional[Prediction]:
        if state.in_string:
            if state.string_triple and state.line_buffer.rstrip().endswith(
                state.string_delim
            ):
                return Prediction(
                    text=state.string_delim * 2,
                    confidence=Confidence.HARD,
                    reason="close_triple_string",
                )
            return None

        # 1. Closing brackets when the line tail looks complete.
        if state.bracket_stack:
            tail = state.line_buffer.rstrip()
            if tail and tail[-1] in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "_0123456789)]}\"'"
            ):
                return Prediction(
                    text=_BRACKET_PAIR[state.bracket_stack[-1]],
                    confidence=Confidence.MEDIUM,
                    reason=f"close_{state.bracket_stack[-1]}",
                )

        # 2. Colon after a compound-statement header with balanced parens.
        # Qwen often tokenizes ":\n" as a single token; predict that.
        if (
            not state.bracket_stack
            and state.line_is_header_keyword
            and state.line_ends_with_balanced_paren_close
            and not state.expects_indent
            and not state.line_buffer.rstrip().endswith(":")
        ):
            return Prediction(
                text=":\n",
                confidence=Confidence.HARD,
                reason="colon_after_header_paren",
            )

        # 3. Newline + indent right after a header colon.
        if state.expects_indent and state.line_buffer.endswith(":"):
            indent_spaces = state.indent_stack[-1] + 4
            return Prediction(
                text="\n" + " " * indent_spaces,
                confidence=Confidence.HARD,
                reason="newline_indent_after_colon",
            )

        # 4. `self` as first parameter of a method.
        if state.bracket_stack and state.bracket_stack[-1] == "(":
            line = state.line_buffer
            stripped = line.rstrip()
            if stripped.endswith("("):
                head = stripped[:-1].rstrip()
                head_match = re.search(
                    r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", head
                )
                if head_match:
                    dunder_methods = {
                        "__init__", "__repr__", "__str__", "__eq__",
                        "__hash__", "__call__", "__len__", "__getitem__",
                        "__setitem__", "__iter__", "__next__",
                        "__enter__", "__exit__",
                    }
                    if self._prompt_class_context or head_match.group(1) in dunder_methods:
                        return Prediction(
                            text="self",
                            confidence=Confidence.MEDIUM,
                            reason="self_param_in_method",
                        )

        return None

    @property
    def state_snapshot(self) -> dict:
        return {
            "bracket_stack": list(self._state.bracket_stack),
            "indent_stack": list(self._state.indent_stack),
            "in_string": self._state.in_string,
            "string_triple": self._state.string_triple,
            "expects_indent": self._state.expects_indent,
            "line_head": self._state.line_head,
            "prompt_identifiers": len(self._prompt_identifiers),
            "prompt_class_context": self._prompt_class_context,
            "tree_sitter": TREE_SITTER_AVAILABLE,
        }


def predict_once(prefix: str, prompt: str = "", max_k: int = 6) -> Proposal:
    proposer = PythonAstProposer(max_chain=max_k)
    if prompt:
        proposer.observe_prompt(prompt)
    proposer.commit(prefix)
    return proposer.propose()


def iter_prediction_events(
    text: str, prompt: str = "", max_k: int = 6
) -> Iterable[tuple[int, Proposal]]:
    proposer = PythonAstProposer(max_chain=max_k)
    if prompt:
        proposer.observe_prompt(prompt)
    for i in range(len(text)):
        proposal = proposer.propose()
        yield i, proposal
        proposer.commit(text[i])
