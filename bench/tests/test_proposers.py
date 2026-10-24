import pytest

from structspec.proposers import (ComboProposer, ContextNgramProposer, LineStartProposer, NgramModelProposer,
                                  SuffixCopyProposer)

CORPUS = [[1, 2, 3, 4, 5, 6] * 3, [7, 8, 9, 1, 2, 3, 4]]


def make_all():
    return [SuffixCopyProposer(), NgramModelProposer(CORPUS), ContextNgramProposer(),
            ComboProposer(SuffixCopyProposer(), [ContextNgramProposer(), NgramModelProposer(CORPUS)])]


@pytest.mark.parametrize("p", make_all(), ids=lambda p: p.name)
def test_respects_max_k(p):
    toks = [9, 9, 1, 2, 3, 4, 5, 6, 1, 2, 3]
    for k in range(0, 6):
        d = p.propose(toks, k)
        assert len(d) <= k
    assert p.propose(toks, 0) == []


@pytest.mark.parametrize("p", make_all(), ids=lambda p: p.name)
def test_empty_tokens(p):
    assert p.propose([], 4) == []
    p.observe([])


@pytest.mark.parametrize("p", make_all(), ids=lambda p: p.name)
def test_deterministic(p):
    toks = [5, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3]
    assert p.propose(toks, 4) == p.propose(list(toks), 4)


def test_suffix_copy_copies_repeated_span():
    p = SuffixCopyProposer(min_match=3, len_base=8)
    toks = [10, 11, 12, 13, 14, 15, 16, 17, 99, 10, 11, 12, 13]
    assert p.propose(toks, 3) == [14, 15, 16]
    assert p.last_match == 4


def test_suffix_copy_draft_grows_with_match_length():
    p = SuffixCopyProposer(min_match=3, len_base=1, len_slope=1.0)
    toks = list(range(20)) + [99] + list(range(8))
    assert p.propose(toks, 16) == list(range(8, 14))  # match 8 -> 1 + (8 - 3) = 6 tokens


def test_suffix_copy_prefers_longest_match():
    p = SuffixCopyProposer(min_match=2)
    toks = [7, 1, 2, 50, 0, 8, 7, 1, 2, 60, 0, 9, 7, 1, 2]
    # [1,2] occurs twice; the longer match [7,1,2] also twice -> most recent wins (60)
    assert p.propose(toks, 1) == [60]
    toks2 = [5, 1, 2, 50, 7, 1, 2, 60, 5, 1, 2]
    assert p.propose(toks2, 1) == [50]


def test_suffix_copy_handles_rewind_to_new_request():
    p = SuffixCopyProposer(min_match=2)
    p.propose([1, 2, 3, 1, 2], 2)
    assert p.propose([4, 5, 6, 4, 5], 1) == [6]


def test_ngram_model_chains_until_confidence_drops():
    m = NgramModelProposer([[1, 2, 3, 4, 5]] * 3 + [[1, 2, 3, 9, 9]], max_order=3, min_order=2, cum_min=0.2)
    assert m.propose([0, 1, 2], 5) == [3, 4, 5]  # P(4|2,3)=0.75 then certain
    strict = NgramModelProposer([[1, 2, 3, 4, 5]] * 3 + [[1, 2, 3, 9, 9]], max_order=3, min_order=2, p_min=0.8)
    assert strict.propose([0, 1, 2], 5) == [3]


def test_ngram_model_ignores_rare_contexts():
    m = NgramModelProposer([[1, 2, 3]], min_count=2)
    assert m.propose([1, 2], 3) == []


def test_combo_uses_copy_on_long_match_else_model():
    model = NgramModelProposer([[40, 41, 42, 43]] * 3)
    c = ComboProposer(SuffixCopyProposer(min_match=3, len_base=8), [ContextNgramProposer(min_count=9), model], copy_min=4)
    long_ctx = [1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4]
    assert c.propose(long_ctx, 2) == [5, 6]
    assert c.propose([9, 40, 41], 2) == [42, 43]


def test_ngram_model_smoothing_penalizes_small_counts():
    corpus = [[1, 2, 3]] * 2 + [[4, 5, 6]] * 20
    m = NgramModelProposer(corpus, max_order=2, min_order=2, p_min=0.8, alpha=2.0)
    assert m.propose([1, 2], 1) == []  # 2/(2+2) = 0.5
    assert m.propose([4, 5], 1) == [6]  # 20/22 = 0.91


def test_context_ngram_completes_identifier_seen_before():
    # "result" = [70, 71]; seen twice in context with the same continuation
    p = ContextNgramProposer(min_count=2, min_order=1)
    toks = [5, 70, 71, 6, 8, 70, 71, 9, 3, 70]
    assert p.propose(toks, 1) == [71]


def test_context_ngram_needs_agreement():
    p = ContextNgramProposer(min_count=2, min_order=1, p_min=0.9)
    toks = [70, 71, 70, 72, 70, 71, 70]  # 70 -> 71 twice, 72 once: 2/3 < 0.9
    assert p.propose(toks, 1) == []


def test_context_ngram_resets_on_new_request():
    p = ContextNgramProposer(min_count=1, min_order=1)
    assert p.propose([1, 2, 1], 1) == [2]
    assert p.propose([3, 4, 3], 1) == [4]


PIECES = ["\n", "    ", "    return", " x", "}", "{", "if", ":", "\n\n", "def f()", " y"]


def test_line_start_predicts_token_after_newline_from_line_features():
    # "def f():\n    return x\n" -> after a line ending ':' the next line starts with token 2
    seq = [9, 7, 0, 2, 3, 0]
    p = LineStartProposer(PIECES, [seq, seq], min_count=2, p_min=0.6)
    assert p.propose([9, 7, 0], 3) == [2]
    assert p.propose([9, 7], 3) == []  # not at a line start


def test_line_start_extends_with_follow_up_proposer():
    seq = [9, 7, 0, 2, 3, 0]
    ext = NgramModelProposer([[0, 2, 3]] * 3, max_order=2, min_order=2)
    p = LineStartProposer(PIECES, [seq, seq], min_count=2, p_min=0.6, extend=ext)
    assert p.propose([9, 7, 0], 3) == [2, 3]
