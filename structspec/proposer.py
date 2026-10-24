"""Draft proposers. `Proposer` is the shared contract from the v2 spec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from .engine import PatternMiner, PythonSyntaxProposer, Rule, propose_draft

INDEX_VERSION = 1


@runtime_checkable
class Proposer(Protocol):
    name: str
    def propose(self, tokens: list[int], max_k: int) -> list[int]: ...
    def observe(self, accepted: list[int]) -> None: ...  # verified tokens only


class ChainProposer:
    """Returns the first non-empty draft; every member observes every commit."""

    name = "chain"

    def __init__(self, proposers: list[Proposer]):
        self.proposers = list(proposers)

    def propose(self, tokens: list[int], max_k: int) -> list[int]:
        for p in self.proposers:
            draft = p.propose(tokens, max_k)
            if draft:
                return draft
        return []

    def observe(self, accepted: list[int]) -> None:
        for p in self.proposers:
            p.observe(accepted)


class RuleProposer:
    """Adapter for the mined-rule miner plus optional Python syntax rules."""

    name = "rules"

    def __init__(self, miner: PatternMiner, syntax: PythonSyntaxProposer | None = None):
        self.miner = miner
        self.syntax = syntax

    def propose(self, tokens: list[int], max_k: int) -> list[int]:
        return propose_draft(self.miner, self.syntax, tokens, max_k)[0]

    def observe(self, accepted: list[int]) -> None:
        pass


class PromptLookupProposer:
    """Copy what followed the most recent earlier occurrence of the current n-gram suffix."""

    name = "prompt_lookup"

    def __init__(self, max_ngram: int = 3, min_ngram: int = 2):
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram

    def propose(self, tokens: list[int], max_k: int) -> list[int]:
        if max_k <= 0:
            return []
        size = len(tokens)
        # shortcut: linear scan per call, fine for a few thousand tokens; use a suffix index for long contexts
        for n in range(min(self.max_ngram, size - 1), self.min_ngram - 1, -1):
            pattern = tokens[-n:]
            for start in range(size - n - 1, -1, -1):
                if tokens[start : start + n] == pattern:
                    return list(tokens[start + n : start + n + max_k])
        return []

    def observe(self, accepted: list[int]) -> None:
        pass


class NgramIndexProposer:
    """Drafts from rules mined offline with the same tokenizer (`structspec index`)."""

    name = "ngram_index"

    def __init__(self, miner: PatternMiner):
        self.miner = miner

    def propose(self, tokens: list[int], max_k: int) -> list[int]:
        return self.miner.propose(tokens, max_k)[0]

    def observe(self, accepted: list[int]) -> None:
        pass


SOURCE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".cs", ".rb", ".php", ".json", ".md",
}
MAX_FILE_BYTES = 1_000_000


def _source_files(paths: list[Path]):
    for root in paths:
        root = Path(root)
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for f in files:
            if f.suffix.lower() in SOURCE_EXTS and f.stat().st_size <= MAX_FILE_BYTES:
                yield f


def build_index(
    model,
    paths: list[Path],
    max_ctx: int = 8,
    min_support: int = 2,
    min_conf: float = 0.9,
    min_rule_ctx: int = 3,
) -> dict:
    seqs = []
    for f in _source_files(paths):
        try:
            seqs.append(model.encode(f.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue
    miner = PatternMiner(max_ctx=max_ctx, min_support=min_support, min_conf=min_conf,
                         min_rule_ctx=min_rule_ctx).fit(seqs)
    return {
        "version": INDEX_VERSION,
        "tokenizer_hash": model.tokenizer_hash(),
        "n_vocab": model.n_vocab,
        "max_ctx": max_ctx,
        "files": len(seqs),
        "rules": [[list(r.ctx), r.token, r.confidence, r.support, r.total, r.tier] for r in miner.rules_by_ctx.values()],
    }


def save_index(index: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index), encoding="utf-8")


def load_index(path: Path, tokenizer_hash: str) -> PatternMiner:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != INDEX_VERSION:
        raise ValueError(f"{path}: unsupported index version {data.get('version')}")
    if data["tokenizer_hash"] != tokenizer_hash:
        raise ValueError(
            f"{path} was built for tokenizer {data['tokenizer_hash']}, but the loaded model's tokenizer is "
            f"{tokenizer_hash}; rebuild it with `structspec index --model <this model>`"
        )
    rules = {tuple(c): Rule(tuple(c), t, conf, sup, tot, tier) for c, t, conf, sup, tot, tier in data["rules"]}
    return PatternMiner(max_ctx=data["max_ctx"]).set_rules(rules)


def default_index_path(tokenizer_hash: str) -> Path:
    return Path.home() / ".cache" / "structspec" / f"index-{tokenizer_hash[:16]}.json"


def default_proposer(model, index: str | None = None, syntax: bool = False) -> ChainProposer:
    """Prompt lookup first, then the tokenizer's n-gram index (explicit path or the default cache file)."""
    proposers: list[Proposer] = [PromptLookupProposer()]
    path = Path(index) if index else default_index_path(model.tokenizer_hash())
    if index or path.exists():
        proposers.append(NgramIndexProposer(load_index(path, model.tokenizer_hash())))
    if syntax:
        proposers.append(RuleProposer(PatternMiner(), PythonSyntaxProposer(model)))
    return ChainProposer(proposers)
