from __future__ import annotations

import random

import pytest
from fake_model import EOG, FakeModel, greedy_continuation

from structspec.llama_backend import SpecStats, run_greedy, run_speculative, speculate
from structspec.proposer import ChainProposer, Proposer

PROMPTS = [[5], [3, 1, 4, 1, 5], [random.Random(7).randrange(1, 50) for _ in range(40)]]
BUDGETS = [1, 7, 64]
MODES = ["truncate", "seq-bonus", "rebuild"]


class Oracle:
    name = "oracle"

    def __init__(self, corrupt_at=None, inject_eog_at=None, overshoot=False):
        self.corrupt_at = corrupt_at
        self.inject_eog_at = inject_eog_at
        self.overshoot = overshoot

    def propose(self, tokens, max_k):
        draft = greedy_continuation(tokens, 50 if self.overshoot else max_k)
        if self.corrupt_at is not None and self.corrupt_at < len(draft):
            draft[self.corrupt_at] = (draft[self.corrupt_at] % 49) + 1
        if self.inject_eog_at is not None and self.inject_eog_at < len(draft):
            draft[self.inject_eog_at] = EOG
        return draft

    def observe(self, accepted):
        pass


class RandomDrafts:
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def propose(self, tokens, max_k):
        return [self.rng.randrange(50) for _ in range(self.rng.randrange(max_k + 1))]

    def observe(self, accepted):
        pass


class Empty:
    name = "empty"

    def propose(self, tokens, max_k):
        return []

    def observe(self, accepted):
        pass


def _check(proposer, prompt, budget, mode, k=4):
    want = run_greedy(FakeModel(), prompt, budget)[0]
    got, _, stats = run_speculative(FakeModel(), proposer, prompt, budget, k=k, reject_mode=mode)
    assert got == want
    return stats


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_oracle_drafts(prompt, budget, mode):
    stats = _check(Oracle(), prompt, budget, mode)
    greedy_passes = run_greedy(FakeModel(), prompt, budget)[2]["passes"]
    if budget > 5 and stats["generated"] > 5:
        assert stats["passes"] < greedy_passes


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_drafts(prompt, budget, mode, seed):
    _check(RandomDrafts(seed), prompt, budget, mode)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("pos", [0, 2])
def test_wrong_token_in_draft(prompt, mode, pos):
    _check(Oracle(corrupt_at=pos), prompt, 64, mode)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_draft_contains_eog(prompt, mode):
    _check(Oracle(inject_eog_at=1), prompt, 64, mode)


def test_greedy_eog_inside_accepted_draft_stops():
    # Find a prompt whose greedy output hits EOG within 40 tokens; the oracle draft then carries it.
    for start in range(1, 200):
        prompt = [start % 49 + 1, start // 49 + 1]
        cont = greedy_continuation(prompt, 40)
        if EOG in cont[2:]:
            break
    else:
        pytest.fail("no prompt reaching EOG")
    got, _, _ = run_speculative(FakeModel(), Oracle(), prompt, 64, k=8)
    want = run_greedy(FakeModel(), prompt, 64)[0]
    assert got == want and got[-1] == EOG


@pytest.mark.parametrize("budget", BUDGETS)
def test_draft_longer_than_budget(budget):
    got, _, _ = run_speculative(FakeModel(), Oracle(overshoot=True), PROMPTS[1], budget)
    assert len(got) - len(PROMPTS[1]) <= budget
    assert got == run_greedy(FakeModel(), PROMPTS[1], budget)[0]


@pytest.mark.parametrize("prompt", PROMPTS)
def test_empty_proposer(prompt):
    _check(Empty(), prompt, 64, "truncate")


def test_observe_gets_only_committed():
    seen = []

    class Recorder(RandomDrafts):
        def observe(self, accepted):
            seen.extend(accepted)

    prompt = PROMPTS[2]
    chunks = list(speculate(FakeModel(), Recorder(3), prompt, 64))
    flat = [t for c in chunks for t in c]
    assert seen == flat == run_greedy(FakeModel(), prompt, 64)[0][len(prompt):]


def test_stats_counts():
    stats = SpecStats()
    chunks = list(speculate(FakeModel(), Oracle(), PROMPTS[1], 32, stats=stats))
    assert stats.generated == sum(len(c) for c in chunks)
    assert stats.accepted <= stats.proposed
    assert stats.passes >= 1


def test_chain_proposer_first_non_empty():
    calls = []

    class Fixed:
        def __init__(self, name, out):
            self.name, self.out = name, out

        def propose(self, tokens, max_k):
            return list(self.out)

        def observe(self, accepted):
            calls.append((self.name, list(accepted)))

    chain = ChainProposer([Fixed("a", []), Fixed("b", [1, 2]), Fixed("c", [3])])
    assert isinstance(chain, Proposer)
    assert chain.propose([9], 4) == [1, 2]
    chain.observe([7])
    assert [c[0] for c in calls] == ["a", "b", "c"]


def test_default_proposer_with_syntax_rules_is_lossless(tmp_path, monkeypatch):
    from structspec.proposer import default_proposer

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    model = FakeModel()
    proposer = default_proposer(model, syntax=True)
    assert [p.name for p in proposer.proposers] == ["suffix_copy", "rules"]
    prompt = model.encode("for i in range(10):\n    if x:\n        pass\n")
    got = run_speculative(FakeModel(), proposer, prompt, 64)[0]
    assert got == run_greedy(FakeModel(), prompt, 64)[0]
