from __future__ import annotations

import numpy as np
from fake_model import FakeModel, greedy_continuation

from structspec.llama_backend import chat_prompt_ids
from structspec.verify import verify_prompt


class Oracle:
    name = "oracle"

    def propose(self, tokens, max_k):
        return greedy_continuation(tokens, max_k)

    def observe(self, accepted):
        pass


PROMPT = [3, 1, 4, 1, 5, 9, 2, 6]


def _flip(margin):
    """Batch rows after the first get a different argmax, winning by `margin`."""

    def noise(i, n, row):
        if n > 1 and i >= 1:
            top = int(np.argmax(row))
            row = row.copy()
            row[(top + 1) % len(row)] = row[top] + margin
        return row

    return noise


class SoftModel(FakeModel):
    """Runner-up logit close to the winner, like a real near-tie."""

    def decode_logits(self, tokens, logits_all):
        rows = super().decode_logits(tokens, logits_all)
        rows = rows * 1.0
        for r in rows:
            top = int(np.argmax(r))
            r[(top + 2) % len(r)] = r[top] - 0.1
        return rows


def test_verify_identical_on_fake():
    r = verify_prompt(FakeModel(), Oracle(), PROMPT, 48)
    assert r["identical"] and r["classification"] == "identical" and r["first_mismatch"] is None


def test_verify_flags_bug():
    r = verify_prompt(FakeModel(batch_noise=_flip(5.0)), Oracle(), PROMPT, 48)
    assert not r["identical"]
    assert r["classification"] == "bug"


def test_verify_flags_numerics():
    r = verify_prompt(SoftModel(batch_noise=_flip(0.05)), Oracle(), PROMPT, 48)
    assert not r["identical"]
    assert r["classification"] == "numerics"
    assert r["greedy_margin"] < 0.5


def test_chat_prompt_disables_thinking():
    class Thinker(FakeModel):
        def chat_template(self):
            return "{% if enable_thinking %}<think>{% endif %}"

    msgs = [{"role": "user", "content": "hi"}]
    plain = FakeModel()
    assert chat_prompt_ids(plain, msgs) == plain.encode(plain.apply_chat_template(msgs))
    t = Thinker()
    assert len(chat_prompt_ids(t, msgs)) > len(chat_prompt_ids(t, msgs, thinking=True))
