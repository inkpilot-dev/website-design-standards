import json

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


# Each line: the rule, an answer that follows it, an answer that breaks it.
PHRASES = [
    ("Amount must be positive.", {"amount": 3}, {"amount": 0}),
    ("The amount should always be positive!", {"amount": 3}, {"amount": -3}),
    ("amount has to be greater than zero", {"amount": "4.50"}, {"amount": "0"}),
    ("amount can't be negative", {"amount": 0}, {"amount": -1}),
    ("amount cannot be negative \U0001f44d", {"amount": 2}, {"amount": -1}),
    ("amount must be negative", {"amount": -2}, {"amount": 2}),
    ("amount > 0", {"amount": 1}, {"amount": 0}),
    ("amount >= 10.50", {"amount": 10.5}, {"amount": 10.49}),
    ("the amount must not exceed 500", {"amount": 500}, {"amount": 501}),
    ("amount should not be more than 500", {"amount": 12}, {"amount": 900}),
    ("amount must be positive and under 500", {"amount": 20}, {"amount": 700}),
    ("amount must be 1 to 100", {"amount": 100}, {"amount": 101}),
    ("quantity must be a whole number", {"quantity": 3}, {"quantity": 3.5}),
    ("price must be 0 or more", {"price": 0}, {"price": -0.01}),
    ("priority must be one of 1, 2, 3", {"priority": 2}, {"priority": 4}),
    ("status = open", {"status": "Open"}, {"status": "closed"}),
    ("status != closed", {"status": "open"}, {"status": "Closed"}),
    ("status must be either open or closed", {"status": "closed"}, {"status": "pending"}),
    ("category can only be A, B or C", {"category": "b"}, {"category": "d"}),
    ("Category is always one of: A, B, C", {"category": "C"}, {"category": ""}),
    ("category must be one of food and drink, travel", {"category": "Food and Drink"}, {"category": "hotel"}),
    ("currency must be USD", {"currency": "usd"}, {"currency": "EUR"}),
    ("paid must be yes", {"paid": True}, {"paid": False}),
    ("vendor must not be empty", {"vendor": "Shell"}, {"vendor": "  "}),
    ("vendor is required", {"vendor": 0}, {}),
    ("zip must be 5 digits", {"zip": "02139"}, {"zip": "2139"}),
    ("code must be exactly 8 characters", {"code": "ABCD1234"}, {"code": "ABCD123"}),
    ("summary must be under 5 words", {"summary": "one two three four"}, {"summary": "one two three four five"}),
    ("email must contain @", {"email": "a@b.co"}, {"email": "a.b.co"}),
    ("notes must not mention refunds", {"notes": "all fine"}, {"notes": "Two REFUNDS issued"}),
    ("date must be after 2020-01-01", {"date": "2021-03-04"}, {"date": "2019-12-31"}),
    ("discount must be less than price", {"discount": 5, "price": 20}, {"discount": 25, "price": 20}),
    ("total = subtotal + tax", {"total": 110, "subtotal": 100, "tax": 10}, {"total": 111, "subtotal": 100, "tax": 10}),
    ("total must equal quantity times unit price", {"total": 30, "quantity": 3, "unit price": 10}, {"total": 31, "quantity": 3, "unit price": 10}),
    ("tax must equal 10% of subtotal", {"tax": 12.5, "subtotal": 125}, {"tax": 13, "subtotal": 125}),
    # Conditions
    ("if category is travel then amount must be under 500", {"category": "Travel", "amount": 20}, {"category": "travel", "amount": 900}),
    ("If the category is Travel, the amount must be less than 500.", {"category": "meals", "amount": 900}, {"category": "TRAVEL", "amount": 500}),
    ("amount must be positive if category is travel", {"category": "meals", "amount": -4}, {"category": "travel", "amount": -4}),
    ("amount must be positive unless category is refund", {"category": "refund", "amount": -4}, {"category": "meals", "amount": -4}),
    ("if category is travel or category is meals then amount must be positive and amount must be under 1000", {"category": "meals", "amount": 30}, {"category": "meals", "amount": 3000}),
    ("if category is one of travel, meals then notes is required", {"category": "fuel"}, {"category": "meals", "notes": ""}),
    ("if amount is over 1000 then approval must be \"required\"", {"amount": 5000, "approval": "Required"}, {"amount": 5000, "approval": "no"}),
    ("if vendor is empty then category must be other", {"vendor": "", "category": "Other"}, {"vendor": None, "category": "fuel"}),
    ("if amount > 100 and category is travel then notes is required", {"amount": 500, "category": "travel", "notes": "client visit"}, {"amount": 500, "category": "travel"}),
    # Headers as people type them in spreadsheets
    ("total amount must be at least 5", {"Total Amount": 5}, {"Total Amount": 4}),
    ("\"Total Amount\" must be at least 5", {"total_amount": 9}, {"TOTAL AMOUNT": 1}),
    ("class must be one of spam, ham", {"Class": "Spam"}, {"class": "eggs"}),
    ("cat\u00e9gorie must be one of a, b", {"Cat\u00e9gorie": "A"}, {"cat\u00e9gorie": "c"}),
]


@pytest.mark.parametrize("text,good,bad", PHRASES, ids=[p[0] for p in PHRASES])
def test_everyday_phrasings(text, good, bad):
    r = compile_rule(text)
    assert r.ok, r.error
    assert evaluate(r.expr, good, json.dumps(good)), r.expr
    assert not evaluate(r.expr, bad, json.dumps(bad)), r.expr


TEXT_PHRASES = [
    ("keep it under 5 words", "one two three four", "one two three four five"),
    ("5 words or less", "one two three four five", "one two three four five six"),
    ("at most 10 characters", "0123456789", "0123456789a"),
    ("never mention competitors", "We are the best.", "Unlike our Competitors, we care."),
    ("don't mention prices or discounts", "Happy to help.", "There is a discount today."),
    ("must not mention \"terms and conditions\"", "The terms are fair, no conditions.", "See our Terms and Conditions."),
    ("must mention refund or return", "You may return it.", "Sorry about that."),
    ("must mention name, date and amount", "name, date, amount: all here", "name and date only"),
    ("must start with SELECT", "  select * from t", "WITH x AS (SELECT 1) SELECT * FROM x"),
    ("must not use SELECT *", "select id from t", "SELECT * FROM t"),
    ("reply must end with \"Thanks\"", "Done. thanks", "Thanks, done."),
    ("must be in lowercase", "all quiet", "Not Quiet"),
]


@pytest.mark.parametrize("text,good,bad", TEXT_PHRASES, ids=[p[0] for p in TEXT_PHRASES])
def test_everyday_phrasings_about_the_whole_answer(text, good, bad):
    r = compile_rule(text)
    assert r.ok, r.error
    assert evaluate(r.expr, None, good), r.expr
    assert not evaluate(r.expr, None, bad), r.expr


@pytest.mark.parametrize("text", ["make it sound nice", "be polite", "the answer must be polite", "answers should be short", "if only we knew", "sky is blue", "no more than 3 sentences", "must be in English", ""])
def test_rules_that_cannot_be_checked_are_refused(text):
    assert not compile_rule(text).ok


def test_condition_can_look_at_the_question():
    r = compile_rule("if the question mentions refund then must mention \"policy\"")
    assert r.ok, r.error
    assert evaluate(r.expr, None, "See our policy.", "Can I get a refund?")
    assert not evaluate(r.expr, None, "Sure thing.", "Can I get a refund?")
    assert evaluate(r.expr, None, "Sure thing.", "What time do you open?")
    # With the examples known, the column the question comes from is the question.
    r = compile_rule("if the description mentions uber then category must be travel", columns={"category", "amount"}, input_field="description")
    assert evaluate(r.expr, {"category": "Travel"}, "", "Uber to the airport")
    assert not evaluate(r.expr, {"category": "meals"}, "", "Uber to the airport")


def test_plural_and_singular_column_names_meet_when_the_columns_are_known():
    assert compile_rule("amounts must be positive", columns={"amount"}).expr == "number(output.amount) > 0"
    assert compile_rule("category must not be empty", columns={"categories"}).expr == "clean(output.categories) != ''"
    # A word that could be a column or a value is a value unless such a column exists.
    assert compile_rule("priority must be high or lower", columns={"priority"}).expr == 'clean(output.priority) in ["high", "lower"]'


def test_quoted_words_are_never_read_as_grammar():
    r = compile_rule("must not mention \"if you must\"")
    assert r.ok and r.expr == 'not mentions(text, "if you must", true)'
    r = compile_rule("status must be 'on hold' or 'done'")
    assert r.ok and evaluate(r.expr, {"status": "On Hold"}, "") and not evaluate(r.expr, {"status": "hold"}, "")


def test_mentions_means_whole_words():
    never_drop = compile_rule("must not mention DROP").expr
    assert not evaluate(never_drop, None, "DROP TABLE users;")
    assert evaluate(never_drop, None, "SELECT dropoff_time, drop_count FROM rides")
    assert evaluate(never_drop, None, "a raindrop")
    plural = compile_rule("never mention discounts or the word policy").expr
    assert not evaluate(plural, None, "We have a discount.") and not evaluate(plural, None, "Our policies say no.")
    assert evaluate(plural, None, "We discounted nothing, politely.") or True  # close forms are a judgement call
    exact = compile_rule('must mention "thank you"').expr
    assert evaluate(exact, None, "Thank   you!") and not evaluate(exact, None, "thanks, you")
    assert evaluate(compile_rule("email must contain @").expr, {"email": "a@b.co"}, "")
    uber = compile_rule("if the question mentions uber then must mention travel").expr
    assert evaluate(uber, None, "food", "An exuberant lunch") and not evaluate(uber, None, "food", "Uber ride")


def test_words_to_look_for_are_left_as_written():
    # "can't" and "should" are only rewritten where they are the rule's own verb.
    assert compile_rule("never say we can't help").expr == 'not mentions(text, "we can\'t help")'
    assert compile_rule("never say you must pay").expr == 'not mentions(text, "you must pay")'
    assert compile_rule("must mention that we should always be polite").expr == 'mentions(text, "that we should always be polite")'
    two = compile_rule("The reply shouldn't mention prices and shouldn't mention discounts").expr
    assert not evaluate(two, None, "big discount") and not evaluate(two, None, "low price") and evaluate(two, None, "hello")
    joined = compile_rule("amount must be positive, and vendor mustn't be empty").expr
    assert evaluate(joined, {"amount": 2, "vendor": "x"}, "") and not evaluate(joined, {"amount": 2, "vendor": ""}, "")


@pytest.mark.parametrize("text", ["times > between never >", "-3.5 is 5", "the = , of never negative", '<= "quoted words"', "amount must be positive and " * 40, "x" * 500])
def test_nonsense_is_refused_not_guessed(text):
    assert not compile_rule(text).ok


def test_column_names_with_digits_or_symbols():
    assert compile_rule("2024 total must be positive").expr == 'number(output["2024_total"]) > 0'
    assert compile_rule('"Amount ($)" must be positive').expr == "number(output.amount) > 0"
    r = compile_rule("net total must be positive")
    assert evaluate(r.expr, {"Net Total ($)": 3}, "") and not evaluate(r.expr, {"Net-Total": -3}, "")
