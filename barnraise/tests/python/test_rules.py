import pytest

from engine.verify.rules import RuleError, compile_rule, evaluate, self_test


@pytest.mark.parametrize(
    "text,good,bad",
    [
        ("category must be one of fuel, meals, software", {"category": "Fuel"}, {"category": "toys"}),
        ("amount must be positive", {"amount": 3.5}, {"amount": -1}),
        ("amount must be between 1 and 100", {"amount": 50}, {"amount": 150}),
        ("total must equal the sum of the line items", {"total": 30, "line_items": [{"amount": 10}, {"amount": 20}]}, {"total": 31, "line_items": [{"amount": 10}, {"amount": 20}]}),
        ("vendor is required", {"vendor": "Shell"}, {"vendor": ""}),
    ],
)
def test_compile_and_eval(text, good, bad):
    r = compile_rule(text)
    assert r.ok, r.error
    assert evaluate(r.expr, good, "")
    assert not evaluate(r.expr, bad, "")


def test_text_rules():
    assert evaluate(compile_rule("no more than 3 words").expr, None, "one two three")
    assert not evaluate(compile_rule("no more than 3 words").expr, None, "one two three four")
    assert not evaluate(compile_rule('must not mention "DROP"').expr, None, "drop table x")
    assert evaluate(compile_rule('must mention "thank you"').expr, None, "Thank you!")


def test_unparseable_rule_gives_guidance():
    r = compile_rule("make it sound nice")
    assert not r.ok and "Try phrasing" in r.error


@pytest.mark.parametrize("expr", ["output.__class__", "__import__('os')", "open('x')", "'a' * 100000000", "[x for x in output for y in output]", "lambda: 1"])
def test_interpreter_rejects_unsafe(expr):
    with pytest.raises(RuleError):
        evaluate(expr, {"a": 1}, "")


def test_self_test_detects_weak_rule():
    refs = [({"category": "fuel", "amount": 10}, "")]
    strong = self_test(compile_rule("amount must be positive").expr, refs)
    assert strong["accepted"] == 1
    weak = self_test("1 == 1", refs)
    assert weak["discrimination"] is None or weak["discrimination"] < 0.8
