import json
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


class PieceTok:
    """A stand-in tokenizer whose token i decodes to pieces[i]."""

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[i] for i in ids)


def live_output(pieces, show, in_think=True):
    import io

    import numpy as np

    from paper_distill.infer import LiveText
    out = io.StringIO()
    live = LiveText(PieceTok(pieces), show, in_think=in_think, out=out)
    live.put(np.array([[0, 1, 2]]))  # the prompt, which must not be printed
    for i in range(len(pieces)):
        live.put(np.array([i]))
    live.end()
    return out.getvalue()


def test_live_text_hides_or_shows_reasoning_and_never_prints_tags():
    answer = "It is four.\n\nFinal answer: 4"
    # the closing tag arrives split across tokens, and character by character
    for pieces in (["<think>", "\n", "Two plus two ", "makes four.", "\n", "</thi", "nk>", "\n\n", "It is four.",
                    "\n\nFinal answer: 4"],
                   list("<think>\nTwo plus two makes four.\n</think>\n\n" + answer)):
        assert live_output(pieces, show=False) == answer
        shown = live_output(pieces, show=True)
        assert shown.endswith("\n\n" + answer) and "Two plus two makes four." in shown
        assert "<think>" not in shown and "</think>" not in shown and "nk>" not in shown.replace("think", "")


def test_live_text_shows_dots_for_long_hidden_reasoning_and_handles_no_close_tag():
    long = ["<think>\n"] + ["step " * 100] * 5 + ["\n</think>\n\n", "Done.\n\nFinal answer: 1"]
    out = live_output(long, show=False)
    assert out.startswith("....") and out.endswith("\n\nDone.\n\nFinal answer: 1") and "step" not in out
    cut_off = ["<think>\n", "still thinking when the budget ran out"]
    assert live_output(cut_off, show=False) == ""
    assert live_output(cut_off, show=True).strip() == "still thinking when the budget ran out"
    assert live_output(["Plain answer, no reasoning."], show=False, in_think=False) == "Plain answer, no reasoning."


def test_chat_loop_commands_and_answers():
    from paper_distill.infer import chat
    lines = iter(["", "/help", "/reasoning", "What is 2+2?", "/vote", "And 3+3?", "/quit", "never reached"])
    printed = []
    chat(Scripted(["4", "6", "6", "6"]), "sys", INF, ask=lambda prompt: next(lines), out=printed.append)
    text = "\n".join(printed)
    assert "reasoning is now shown" in text and "majority voting is now on" in text
    assert "some steps" in text and "Final answer: 4" in text and "Final answer: 6" in text
    assert "3 samples, 100% agreed" in text and "never reached" not in text


def test_chat_loop_ends_quietly_on_ctrl_c_or_end_of_input():
    from paper_distill.infer import chat

    def interrupted(prompt):
        raise KeyboardInterrupt

    chat(Scripted([]), "sys", INF, ask=interrupted, out=lambda *_: None)  # must simply return


def test_retry_only_touches_unanswered_problems_and_summary_pairs_runs(tmp_path, capsys):
    from paper_distill import evaluate

    problems = {str(i): {"id": str(i), "question": f"q{i}", "answer_type": "int", "ground_truth": str(i), "meta": {}}
                for i in range(5)}
    rows = [{"id": "0", "domain": "d", "difficulty": 1, "final": "0", "correct": True},
            {"id": "1", "domain": "d", "difficulty": 1, "final": "9", "correct": False},   # wrong, but answered
            {"id": "2", "domain": "d", "difficulty": 2, "final": None, "correct": False},
            {"id": "3", "domain": "d", "difficulty": 2, "final": None, "correct": False},
            {"id": "4", "domain": "d", "difficulty": 2, "final": None, "correct": False}]

    class Second:
        """Answers problem 2 on the first retry, problem 3 on the second, and never answers problem 4."""
        calls = []

        def ask(self, system, questions, n=1, **gen):
            Second.calls.append(list(questions))
            out = []
            for q in questions:
                ok = q == "q2" or (q == "q3" and len(Second.calls) == 2)
                out.append([reply(q[1:]) if ok else "<think>\nround and round\n</think>\n\nNot sure."])
            return out

    res = evaluate.retry_unanswered(Second(), "sys", {"rows": rows}, problems, INF)
    assert Second.calls == [["q2", "q3", "q4"], ["q3", "q4"]]
    assert [r["correct"] for r in res["rows"]] == [True, False, True, True, False]
    assert res["overall"]["correct"] == 3 and res["no_answer"] == 1
    assert rows[2]["final"] is None                      # the saved run itself is not altered

    out = tmp_path / "out"
    out.mkdir()
    base = {"label": "base", "set": "procedural", "strategy": "single", "no_answer": 3, **evaluate.tabulate(rows)}
    mine = {"label": "paper", "set": "procedural", "strategy": "single", "no_answer": 1, **res}
    (out / "eval_base_procedural_single.json").write_text(json.dumps(base))
    (out / "eval_paper_procedural_single.json").write_text(json.dumps(mine))
    evaluate.summary({"root": tmp_path})
    text = capsys.readouterr().out
    assert "base alone was right on 0, paper alone on 2; they agreed on the other 3" in text
