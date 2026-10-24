import math

import pytest

from bench.replay import predict, replay

TRACE = {"prompt_ids": [1, 2, 3], "out_ids": list(range(10, 30))}


class Oracle:
    name = "oracle"

    def __init__(self, out, n_prompt):
        self.out, self.n_prompt, self.seen = out, n_prompt, []

    def propose(self, tokens, max_k):
        i = len(tokens) - self.n_prompt
        return self.out[i:i + max_k]

    def observe(self, accepted):
        self.seen += accepted


class Null:
    name = "null"

    def propose(self, tokens, max_k):
        return []

    def observe(self, accepted):
        pass


class Wrong(Oracle):
    def propose(self, tokens, max_k):
        return [-1] * max_k


def flat_cost(batch, ctx):
    return 10.0 + batch


def test_replay_oracle():
    r = replay(Oracle(TRACE["out_ids"], 3), TRACE, max_k=4)
    assert r["passes"] == math.ceil(20 / 5)
    assert r["accepted"] == 16 and r["n_out"] == 20


def test_replay_null():
    r = replay(Null(), TRACE, max_k=4)
    assert r["passes"] == 20 and r["drafted"] == 0 and r["steps_with_draft"] == 0


def test_observe_gets_only_verified():
    w = Wrong(TRACE["out_ids"], 3)
    replay(w, TRACE, max_k=3)
    assert w.seen == TRACE["out_ids"]
    o = Oracle(TRACE["out_ids"], 3)
    replay(o, TRACE, max_k=3)
    assert o.seen == TRACE["out_ids"]


def test_predict_uses_batch_cost():
    null = replay(Null(), TRACE, max_k=4, cost=flat_cost)
    assert predict(null) == pytest.approx(1.0, rel=1e-3)  # proposer time counts too
    orc = replay(Oracle(TRACE["out_ids"], 3), TRACE, max_k=4, cost=flat_cost)
    assert predict(orc) == pytest.approx((20 * 11) / (4 * 15), rel=1e-3)
    wrong = replay(Wrong(TRACE["out_ids"], 3), TRACE, max_k=4, cost=flat_cost)
    assert predict(wrong) < 1.0
