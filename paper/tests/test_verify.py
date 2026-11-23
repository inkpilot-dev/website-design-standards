import pytest

from paper_distill.verify import canonical, check_answer, extract_final_answer, numeric_value

OK = [
    ("int", "42", "42"), ("int", "42", "42 crates"), ("int", "1250", "1,250"), ("int", "-8", "y = −8"),
    ("int", "7", "\\boxed{7}"), ("int", "12", "**12**"), ("int", "12", "12.0"), ("int", "6665", "a_8 = 6665"),
    ("fraction", "770/37", "770/37 hours"), ("fraction", "5/2", "\\frac{5}{2}"), ("fraction", "5/2", "2 1/2"),
    ("fraction", "1/4", "0.25"), ("fraction", "3", "3"), ("fraction", "3/4", "0.75 or 3/4"),
    ("number", "35.64", "$35.64"), ("number", "35.60", "35.6"), ("number", "250.00", "250"),
    ("yesno", "Yes", "Yes."), ("yesno", "No", "No, it does not follow"), ("yesno", "Yes", "yes"),
    ("list", "Callum, Felix, Lucia", "Callum, Felix, and Lucia"), ("list", "A, B, F", "A → B → F"),
    ("list", "A, B, F", "A-B-F"), ("set", "Chloe, Omar", "Omar and Chloe"),
    ("time", "15:40", "15:40"), ("time", "15:40", "3:40 PM"), ("time", "09:05", "9:05"),
    ("exact", "[8, 9, 10]", "`[8,9,10]`"), ("exact", "a 5 6", "a 5 6"), ("exact", "[9, 5] 2", "[9, 5] 2"),
    ("word", "vnpr", '"vnpr"'), ("word", "vnpr", "VNPR"),
    ("choice", "B", "(B)"), ("choice", "B", "B) 42"), ("choice", "C", "Option C"),
]
BAD = [
    ("int", "42", "43"), ("int", "42", "42 or 43"), ("int", "150", "2 hours 30 minutes"), ("int", "5", ""),
    ("int", "5", "15:05"), ("fraction", "1/3", "0.3333"), ("fraction", "5/2", "5/3"),
    ("number", "35.64", "35.6"), ("yesno", "Yes", "No"), ("yesno", "Yes", "It depends"),
    ("list", "A, B, F", "A, F, B"), ("set", "Chloe, Omar", "Chloe"), ("set", "Chloe, Omar", "Chloe, Omar, Hana"),
    ("time", "15:40", "15:45"), ("time", "15:40", "3:40 AM"), ("exact", "3 5", "35"), ("exact", "[8, 9]", "[9, 8]"),
    ("word", "vnpr", "the string is vnpr"), ("word", "vnpr", "vnrp"), ("choice", "B", "C"),
]


@pytest.mark.parametrize("kind,truth,cand", OK)
def test_accepts(kind, truth, cand):
    assert check_answer(kind, truth, cand)


@pytest.mark.parametrize("kind,truth,cand", BAD)
def test_rejects(kind, truth, cand):
    assert not check_answer(kind, truth, cand)


def test_text_with_distractors():
    meta = {"distractors": ["Felix", "Lucia"]}
    assert check_answer("text", "Callum", "Callum", meta)
    assert check_answer("text", "Callum", "It is Callum.", meta)
    assert not check_answer("text", "Callum", "Callum or Felix", meta)
    assert not check_answer("text", "Callum", "Felix", meta)
    assert check_answer("text", "silver key", "The silver key", {"distractors": ["white mug"]})
    assert check_answer("text", "Wednesday", "wednesday")
    assert not check_answer("text", "Wednesday", "It will be Wednesday")  # no distractors: exact only


def test_extract():
    assert extract_final_answer("Work.\n\nFinal answer: 42") == "42"
    assert extract_final_answer("Work.\n**Final answer:** 42\n") == "42"
    assert extract_final_answer("Work.\nFinal Answer: **3/4**") == "3/4"
    assert extract_final_answer("Final answer: 42\nMore text after.") is None
    assert extract_final_answer("Final answer: 42\nMore text after.", strict=False) == "42"
    assert extract_final_answer("So \\boxed{7}.", strict=False) == "7"
    assert extract_final_answer("no answer here", strict=False) is None
    assert extract_final_answer("Final answer:") is None


def test_numeric_and_canonical():
    assert numeric_value("3-5") is None
    assert numeric_value("x = -5") == -5
    assert canonical("42 apples") == canonical("42.0")
    assert canonical("Yes.") == canonical("yes")
    assert canonical("Callum") != canonical("Felix")
    assert canonical("") is None
