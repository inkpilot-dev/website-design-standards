"""End-to-end test of the build stages against a scripted stand-in for the teacher.

The stand-in exists only so the pipeline's bookkeeping can be tested without a GPU.
Nothing it produces is ever shipped as data.
"""
import json
import re
import shutil
from pathlib import Path

import pytest

from paper_distill import build
from paper_distill.config import load, read_jsonl
from paper_distill.schema import validate_record
from paper_distill.teacher import Reply, split_reasoning
from paper_distill.topics import parse_authored

ROOT = Path(__file__).resolve().parents[1]
THINK = "First I restate what is asked and list the quantities involved. Then I work through each step in turn and check the result against the conditions given."


class ScriptedTeacher:
    """Answers generated problems from an answer sheet, with a fixed share of bad replies."""

    def __init__(self, sheet):
        self.sheet, self.n, self.authored = sheet, 0, 0

    def chat(self, messages, *, max_tokens=None, direct=False, temperature=None):
        text = messages[-1]["content"]
        if text.startswith("Write "):
            n = int(re.search(r"Write (\d+)", text).group(1))
            items = []
            for _ in range(n):
                self.authored += 1
                k = self.authored
                q = (f"Depot {k} holds {k + 3} crates and a truck brings {2 * k + 1} more crates in the morning, "
                     f"before {k % 3} crates are sent away at batch {k * 7919}. How many crates are in the depot?")
                self.sheet[q] = str(k + 3 + 2 * k + 1 - k % 3)
                items.append({"question": q, "answer": self.sheet[q], "answer_type": "number"})
            return Reply("", "Here you go:\n" + json.dumps(items), "stop", 50, 0.01)
        question = text.split("\n\n", 1)[1]
        self.n += 1
        answer = self.sheet[question]
        if self.n % 5 == 0:
            return Reply(THINK, "It works out.\n\nFinal answer: definitely not this", "stop", 40, 0.01)
        if self.n % 7 == 0:
            return Reply(THINK, f"Cut off before the end\n\nFinal answer: {answer}", "length", 40, 0.01)
        if self.n % 11 == 0:
            return Reply("", f"Final answer: {answer}", "stop", 5, 0.01)  # no reasoning
        return Reply("" if direct else THINK, f"The steps give the result.\n\nFinal answer: {answer}", "stop", 40, 0.01)


@pytest.fixture
def cfg(tmp_path):
    shutil.copy(ROOT / "config.yaml", tmp_path / "config.yaml")
    c = load(tmp_path / "config.yaml")
    c["dataset"].update(target_size=300, min_size=250, eval_holdout=40, oversample=2.0)
    c["authoring"]["batch_size"] = 5
    c["teacher"]["concurrency"] = 3
    return c


def run(cfg):
    build.prepare(cfg)
    p = build.paths(cfg)
    sheet = {s["question"]: s["ground_truth"] for s in read_jsonl(p["procedural"])}
    teacher = ScriptedTeacher(sheet)
    build.author(cfg, teacher)
    build.solve(cfg, teacher)
    return teacher, build.select(cfg)


def test_pipeline_end_to_end(cfg):
    teacher, report = run(cfg)
    p = build.paths(cfg)
    train, val = read_jsonl(p["train"]), read_jsonl(p["val"])
    rows = train + val
    assert len(rows) == 300 and len(val) == 9
    assert all(validate_record(r) == [] for r in rows)
    assert report["by_source"] == {"procedural": 180, "teacher_authored": 120}
    assert max(report["by_domain"].values()) <= 42  # 14% cap
    # every kept answer really is the right one, and none of the scripted bad replies got through
    seeds = {s["id"]: s for src in build.SOURCES for s in read_jsonl(p[src])}
    for r in rows:
        assert r["reasoning"] == THINK and r["verification"]["reference"] == seeds[r["id"]]["ground_truth"]
        assert "definitely not" not in r["answer"] and "Cut off" not in r["answer"]
    traces = read_jsonl(p["traces"])
    reasons = {x for t in traces for x in t["reasons"]}
    assert {"wrong_answer", "truncated", "reasoning_too_short"} <= reasons
    assert not {r["id"] for r in rows} & {r["id"] for r in read_jsonl(p["eval"])}
    assert build.audit(cfg)


def test_resume_makes_no_new_calls_and_same_output(cfg):
    teacher, _ = run(cfg)
    p = build.paths(cfg)
    before, calls = p["train"].read_text(), teacher.n
    build.solve(cfg, teacher)  # nothing left to do
    build.select(cfg)
    assert teacher.n == calls and p["train"].read_text() == before


def test_limit_and_interruption(cfg):
    build.prepare(cfg)
    p = build.paths(cfg)
    sheet = {s["question"]: s["ground_truth"] for s in read_jsonl(p["procedural"])}
    teacher = ScriptedTeacher(sheet)
    cfg["dataset"]["mix"] = {"procedural": 1.0, "teacher_authored": 0.0, "public": 0.0}
    stats = build.solve(cfg, teacher, limit=25)
    assert stats["calls"] == 25 and len(read_jsonl(p["traces"])) == 25
    build.solve(cfg, teacher)  # carries on from the 25 already done
    ids = [t["seed_id"] for t in read_jsonl(p["traces"]) if t["ok"]]
    assert len(ids) == len(set(ids)) >= 300


def test_audit_catches_a_broken_record(cfg):
    run(cfg)
    p = build.paths(cfg)
    rows = read_jsonl(p["train"])
    rows[0]["answer"] = "No final line here"
    rows[1]["reasoning"] = "has a </think> tag"
    p["train"].write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert not build.audit(cfg)


def test_teacher_down_stops_the_run(cfg):
    build.prepare(cfg)

    class Down:
        def chat(self, *a, **k):
            raise RuntimeError("teacher unreachable")

    with pytest.raises(RuntimeError):
        build.solve(cfg, Down())
    assert read_jsonl(build.paths(cfg)["traces"]) == []


def test_split_reasoning_formats():
    assert split_reasoning({"content": "Answer", "reasoning": "why"}) == ("why", "Answer")  # Ollama
    assert split_reasoning({"content": "Answer", "reasoning_content": "why"}) == ("why", "Answer")  # llama.cpp
    assert split_reasoning({"content": "<think>\nwhy\n</think>\n\nAnswer"}) == ("why", "Answer")  # inline
    assert split_reasoning({"content": "why\n</think>\nAnswer"}) == ("why", "Answer")  # tag opened in the prompt
    assert split_reasoning({"content": "<think>\n\n</think>\n\nAnswer"}) == ("", "Answer")
    assert split_reasoning({"content": None}) == ("", "")


def test_parse_authored_drops_malformed():
    good = {"question": "A shop sells pens at 3 for $2. How much do 12 pens cost in dollars?", "answer": 8,
            "answer_type": "number"}
    choice = {"question": "Which is largest of these four numbers? A) 3 B) 9 C) 4 D) 1", "answer": "B",
              "answer_type": "choice"}
    bad = [{"question": "too short", "answer": "1", "answer_type": "number"},
           {"question": "A long enough question without any options listed at all, is it?", "answer": "B",
            "answer_type": "choice"},
           {"question": "A long enough question whose answer is not a number at all, right?", "answer": "many",
            "answer_type": "number"}, "not a dict"]
    out = parse_authored("Sure!\n" + json.dumps([good, choice] + bad))
    assert [o["answer"] for o in out] == ["8", "B"]
    assert parse_authored("no json here") == []
    assert len(parse_authored(json.dumps(good) + "\n" + json.dumps(choice))) == 2  # not wrapped in an array
