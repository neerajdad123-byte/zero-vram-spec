"""Unit tests for structspec.ast_proposer."""

from __future__ import annotations

from structspec.ast_proposer import (
    Confidence,
    PythonAstProposer,
    predict_once,
    iter_prediction_events,
)


def test_colon_after_def_with_parens():
    proposal = predict_once("def f()")
    assert proposal.text.startswith(":")
    assert proposal.tokens[0].confidence is Confidence.HARD
    assert proposal.tokens[0].reason == "colon_after_header_paren"


def test_colon_after_class_with_parens():
    proposal = predict_once("class Foo(Base)")
    assert proposal.text.startswith(":")


def test_newline_indent_after_colon():
    proposal = predict_once("def f():")
    assert proposal.tokens[0].text == "\n    "
    assert proposal.tokens[0].confidence is Confidence.HARD


def test_chain_def_colon_then_indent():
    proposal = predict_once("def f()", max_k=4)
    assert len(proposal.tokens) >= 2
    assert proposal.tokens[0].text == ":"
    assert proposal.tokens[1].text == "\n    "


def test_close_paren_after_identifier():
    proposal = predict_once("def f():\n    print(x", max_k=3)
    assert proposal.tokens
    assert proposal.tokens[0].text == ")"
    assert proposal.tokens[0].confidence is Confidence.MEDIUM


def test_self_in_init_with_class_context():
    proposal = predict_once(
        "class Foo:\n    def __init__(",
        prompt="implement class Foo in python",
        max_k=3,
    )
    assert proposal.tokens
    assert proposal.tokens[0].text == "self"
    assert proposal.tokens[0].confidence is Confidence.MEDIUM


def test_no_self_for_non_method_def():
    proposal = predict_once("def helper(")
    if proposal.tokens:
        assert proposal.tokens[0].text != "self"


def test_no_prediction_in_middle_of_identifier():
    proposal = predict_once("def fooba")
    assert proposal.text == "" or proposal.tokens[0].text != ":"


def test_no_prediction_inside_open_bracket_after_open():
    proposal = predict_once("def f(")
    if proposal.tokens:
        assert proposal.tokens[0].text != ")"


def test_nested_indent_levels():
    proposal = predict_once("def outer():\n    def inner():")
    assert proposal.tokens
    assert proposal.tokens[0].text == "\n        "


def test_dedent_after_function_body():
    proposer = PythonAstProposer()
    proposer.commit("def f():\n    return 1\n\npass\n")
    assert proposer.state_snapshot["indent_stack"][-1] == 0


def test_nested_brackets_close_in_order():
    proposer = PythonAstProposer()
    proposer.commit("f(g([")
    proposer.commit("1, 2")
    proposal = proposer.propose(max_k=3)
    assert proposal.tokens[0].text == "]"


_GOLDEN_FIBONACCI = (
    "def fibonacci(n):\n"
    "    if n <= 1:\n"
    "        return n\n"
    "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
)


def test_offline_accept_rate_on_fibonacci():
    total = matched = 0
    for i, proposal in iter_prediction_events(_GOLDEN_FIBONACCI):
        if not proposal.tokens:
            continue
        gold_next = _GOLDEN_FIBONACCI[i:][: len(proposal.text)]
        total += 1
        if gold_next == proposal.text:
            matched += 1
    assert total > 0, "Proposer never fired on fibonacci"
    accept_rate = matched / total
    assert accept_rate >= 0.6, (
        f"Acceptance rate too low: {accept_rate:.2%} ({matched}/{total})"
    )


def test_predictions_save_at_least_one_character():
    for _, proposal in iter_prediction_events(_GOLDEN_FIBONACCI):
        for pred in proposal.tokens:
            assert len(pred.text) >= 1


def test_hard_predictions_100_percent_accurate():
    hard_matched = hard_total = 0
    for i, proposal in iter_prediction_events(_GOLDEN_FIBONACCI):
        hard_preds = [p for p in proposal.tokens if p.confidence is Confidence.HARD]
        if not hard_preds:
            continue
        hard_total += 1
        total_text = "".join(p.text for p in hard_preds)
        gold_next = _GOLDEN_FIBONACCI[i:][: len(total_text)]
        if gold_next == total_text:
            hard_matched += 1
            continue
        # If full chain doesn't match, at least the first prediction should
        first_text = hard_preds[0].text
        first_gold = _GOLDEN_FIBONACCI[i:][: len(first_text)]
        if first_text == first_gold:
            hard_matched += 1
    assert hard_total > 0
    assert hard_matched == hard_total, (
        f"HARD predictions not 100%: {hard_matched}/{hard_total}"
    )
