"""Voting logic, tested with a scripted backend (no model needed)."""
from paper_distill.infer import parse, solve, tally
from paper_distill.train import lr_factor, make_batches

INF = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 100, "think_budget": 50, "votes": 5, "early_stop": 3}


def reply(ans):
    return f"<think>\nsome steps\n</think>\n\nSo it is.\n\nFinal answer: {ans}"


class Scripted:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def ask(self, system, questions, n=1, **gen):
        self.calls.append(n)
        return [[reply(self.answers.pop(0)) for _ in range(n)] for _ in questions]


def test_stops_early_when_first_three_agree():
    b = Scripted(["42", "42.0", "42 apples", "7", "7"])
    r = solve(b, "sys", "q", INF, "vote")
    assert b.calls == [3] and r["samples"] == 3 and r["agreement"] == 1.0 and r["final_answer"] == "42"


def test_draws_more_on_disagreement_and_takes_majority():
    b = Scripted(["41", "42", "7", "42", "42"])
    r = solve(b, "sys", "q", INF, "vote")
    assert b.calls == [3, 2] and r["samples"] == 5 and r["final_answer"] == "42" and r["agreement"] == 0.6


def test_tie_goes_to_first_seen_and_single_makes_one_call():
    b = Scripted(["9", "8", "9", "8", "7"])
    assert solve(b, "sys", "q", INF, "vote")["final_answer"] == "9"
    b = Scripted(["5"])
    r = solve(b, "sys", "q", INF, "single")
    assert b.calls == [1] and r["final_answer"] == "5" and r["reasoning"] == "some steps"


def test_no_parsable_answer_still_returns_text():
    class NoAnswer:
        def ask(self, system, questions, n=1, **gen):
            return [["<think>\nhm\n</think>\n\nI am not sure." for _ in range(n)] for _ in questions]
    r = solve(NoAnswer(), "sys", "q", INF, "vote")
    assert r["final_answer"] is None and r["answer"] == "I am not sure." and r["agreement"] == 0.0


def test_parse_and_tally():
    assert parse(reply("3/4")) == ("some steps", "So it is.\n\nFinal answer: 3/4", "3/4")
    votes, first = tally([reply("0.75"), reply("3/4"), reply("1")])
    assert sorted(votes.values()) == [1, 2]


def test_batches_respect_token_budget_and_cover_everything():
    import random
    lengths = [random.Random(i).randint(50, 2000) for i in range(500)]
    batches = make_batches(lengths, 4096, random.Random(0))
    assert sorted(i for b in batches for i in b) == list(range(500))
    assert all(len(b) * max(lengths[i] for i in b) <= 4096 or len(b) == 1 for b in batches)


def test_lr_schedule_warms_up_then_decays_to_floor():
    assert lr_factor(0, 100, 10, 0.1) == 0.1
    assert lr_factor(9, 100, 10, 0.1) == 1.0
    assert abs(lr_factor(100, 100, 10, 0.1) - 0.1) < 1e-9
    assert lr_factor(40, 100, 10, 0.1) > lr_factor(80, 100, 10, 0.1)


def test_parse_handles_reopened_reasoning():
    text = "<think>\nfirst\n</think>\n\nwait, more thought\n</think>\n\nDone.\n\nFinal answer: 2"
    reasoning, answer, final = parse(text)
    assert final == "2" and "</think>" not in answer and "first" in reasoning and "more thought" in reasoning
