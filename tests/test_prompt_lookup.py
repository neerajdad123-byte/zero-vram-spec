from __future__ import annotations

import pytest
from fake_model import FakeModel

from structspec.llama_backend import run_greedy, run_speculative
from structspec.proposer import PromptLookupProposer


def test_prompt_lookup_copies_continuation():
    assert PromptLookupProposer().propose([1, 2, 3, 4, 5, 9, 1, 2, 3], 3) == [4, 5, 9]


def test_prompt_lookup_prefers_longest_and_most_recent():
    p = PromptLookupProposer()
    # [9,1,2] (n=3) wins over the more recent [1,2] match
    assert p.propose([9, 1, 2, 7, 3, 1, 2, 8, 9, 1, 2], 1) == [7]
    # among n=2 matches, the most recent one wins
    assert p.propose([1, 2, 7, 1, 2, 8, 5, 1, 2], 2) == [8, 5]


def test_prompt_lookup_no_match_returns_empty():
    assert PromptLookupProposer().propose([1, 2, 3, 4], 4) == []
    assert PromptLookupProposer().propose([], 4) == []


def test_prompt_lookup_respects_max_k():
    tokens = [1, 2, 3, 4, 5, 6, 7, 1, 2]
    assert PromptLookupProposer().propose(tokens, 2) == [3, 4]
    assert PromptLookupProposer().propose(tokens, 0) == []


@pytest.mark.parametrize("mode", ["truncate", "seq-bonus"])
def test_lossless_with_prompt_lookup(mode):
    model = FakeModel()
    prompt = model.encode("def add(a, b):\n    return a + b\n\ndef add(a, b):\n    return a")
    want = run_greedy(FakeModel(), prompt, 64)[0]
    got, _, _ = run_speculative(model, PromptLookupProposer(), prompt, 64, reject_mode=mode)
    assert got == want
