"""Draft proposers. `Proposer` is the shared contract from the v2 spec."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .engine import PatternMiner, PythonSyntaxProposer, propose_draft


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
