"""End-to-end test of the build stages against a scripted stand-in for the teacher.

The stand-in exists only so the pipeline's bookkeeping can be tested without a GPU.
Nothing it produces is ever shipped as data.
"""
import json
import re
import shutil
import threading
import time
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
        self.sheet, self.n, self.authored, self.limits = sheet, 0, 0, []
        self.lock, self.active, self.peak = threading.Lock(), 0, 0

    def chat(self, messages, **kw):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.001)  # long enough for requests to overlap
            with self.lock:
                return self._reply(messages, **kw)
        finally:
            with self.lock:
                self.active -= 1

    def _reply(self, messages, *, max_tokens=None, direct=False, temperature=None, reasoning_limit=None):
        self.limits.append(reasoning_limit)
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
    c["dataset"].update(target_size=300, min_size=250, eval_holdout=40, oversample=2.0, max_attempts=2,
                        mix={"procedural": 0.6, "teacher_authored": 0.4, "public": 0.0})
    c["authoring"].update(batch_size=5, oversample=2.0)
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
    units = [{**good, "answer": a} for a in ("2.5 cups", "25%", "1,250 dollars")]
    assert [o["answer"] for o in parse_authored(json.dumps(units))] == ["2.5", "25", "1250"]
    assert parse_authored("no json here") == []
    assert len(parse_authored(json.dumps(good) + "\n" + json.dumps(choice))) == 2  # not wrapped in an array


def test_reasoning_caps_follow_difficulty(cfg):
    q = cfg["quality"]
    easy = {"difficulty": 2, "answer_type": "int", "ground_truth": "4", "meta": {}}
    hard = {**easy, "difficulty": 5}
    assert build.reasoning_cap(easy, q) == q["max_reasoning_words"] < build.reasoning_cap(hard, q)
    steps = [f"Step {i}: take value {i} and combine it with {i * 7} to get {i * 8}." for i in range(1, 90)]
    long_think = "\n\n".join(steps)  # realistic shape: many short, different steps
    assert q["max_reasoning_words"] < len(long_think.split()) < q["max_reasoning_words_hard"]
    answer = "It is four.\n\nFinal answer: 4"
    assert build.judge(easy, long_think, answer, "stop", q) == (False, ["reasoning_too_long"], "4")
    assert build.judge(hard, long_think, answer, "stop", q)[0] is True
    # a reply stopped early for length, or cut off, reports that one reason and nothing else
    assert build.judge(easy, long_think, "", "reasoning_limit", q) == (False, ["reasoning_too_long"], None)
    assert build.judge(easy, THINK, "", "length", q) == (False, ["truncated"], None)
    assert build.judge(easy, "", "", "timeout", q) == (False, ["timed_out"], None)


def test_solver_passes_the_cap_to_the_teacher(cfg):
    teacher, _ = run(cfg)
    q = cfg["quality"]
    assert set(teacher.limits) == {None, q["max_reasoning_words"], q["max_reasoning_words_hard"]}  # None: authoring


def seed(i, gen, difficulty=1):
    return {"id": f"{gen}{i}", "source": "procedural", "generator": gen, "difficulty": difficulty}


def test_scheduler_does_not_pour_calls_into_a_failing_group():
    seeds = [seed(i, g) for g in ("easy", "hopeless") for i in range(400)]
    sched = build.Scheduler(seeds, [], {"procedural": 100, "teacher_authored": 0, "public": 0}, 1)
    calls = {"easy": 0, "hopeless": 0}
    while (s := sched.next()) is not None:
        calls[s["generator"]] += 1
        sched.done(s, s["generator"] == "easy")
    assert sched.acc_src["procedural"] == 100
    assert calls["hopeless"] == build.Scheduler.GIVE_UP_AFTER and sched.skipped > 0


def test_scheduler_shares_calls_by_yield_not_only_by_accepted_count():
    seeds = [seed(i, g) for g in ("good", "poor") for i in range(2000)]
    sched = build.Scheduler(seeds, [], {"procedural": 300, "teacher_authored": 0, "public": 0}, 1)
    calls, n = {"good": 0, "poor": 0}, 0
    while (s := sched.next()) is not None:
        calls[s["generator"]] += 1
        n += 1
        sched.done(s, True if s["generator"] == "good" else n % 5 == 0)  # the poor group keeps about 1 in 5
    assert sched.acc_grp[("procedural", "poor")] > 30         # still represented
    assert calls["poor"] < 2.5 * calls["good"]                 # but not at five times the cost


def test_scheduler_resumes_its_bookkeeping_from_the_log():
    seeds = [seed(i, "g", 4) for i in range(20)]
    traces = [{"seed_id": f"g{i}", "ok": False} for i in range(build.Scheduler.GIVE_UP_AFTER)]
    sched = build.Scheduler(seeds, traces, {"procedural": 10, "teacher_authored": 0, "public": 0}, 1)
    assert sched.next() is None and sched.skipped == 20 - build.Scheduler.GIVE_UP_AFTER


def test_select_keeps_long_traces_to_a_minority(cfg):
    cfg["dataset"]["mix"] = {"procedural": 1.0, "teacher_authored": 0.0, "public": 0.0}
    cfg["dataset"]["max_long_share"] = 0.1
    cfg["quality"]["max_reasoning_words"] = 20  # the scripted reasoning is 27 words, so hard problems count as long
    cfg["quality"]["max_reasoning_words_hard"] = 60
    build.prepare(cfg)
    p = build.paths(cfg)
    sheet = {s["question"]: s["ground_truth"] for s in read_jsonl(p["procedural"])}
    build.solve(cfg, ScriptedTeacher(sheet))
    build.select(cfg)
    rows = read_jsonl(p["train"]) + read_jsonl(p["val"])
    assert rows and all(r["difficulty"] >= 4 for r in rows)  # easy ones were over their cap and rejected
    assert len(rows) <= 30                                    # 10% of the 300 target


def test_stats_and_pilot_report(cfg, capsys):
    build.prepare(cfg)
    p = build.paths(cfg)
    build.stats(cfg)
    assert "no teacher answers" in capsys.readouterr().out
    sheet = {s["question"]: s["ground_truth"] for s in read_jsonl(p["procedural"])}
    build.pilot(cfg, ScriptedTeacher(sheet), calls=40, author_calls=2)
    out = capsys.readouterr().out
    assert "pilot: generated problems:" in out and "pilot: teacher-written questions:" in out
    assert "pilot: projected time for 300 records" in out and "Kept by difficulty" in out
    traces = read_jsonl(p["traces"])
    assert len(traces) == 40 + 10  # 40 generated, plus the 10 questions the scripted teacher wrote
    build.stats(cfg)
    out = capsys.readouterr().out
    assert "procedural" in out and "teacher_authored" in out and "total:" in out


def test_ctrl_c_stops_cleanly_and_the_run_resumes(cfg, monkeypatch):
    import threading
    import time as real_time
    build.prepare(cfg)
    p = build.paths(cfg)
    sheet = {s["question"]: s["ground_truth"] for s in read_jsonl(p["procedural"])}
    cfg["dataset"]["mix"] = {"procedural": 1.0, "teacher_authored": 0.0, "public": 0.0}
    teacher = ScriptedTeacher(sheet)
    real_sleep, state = real_time.sleep, {"fired": False}

    def sleep(seconds):
        if threading.current_thread() is threading.main_thread() and not state["fired"] and teacher.n >= 30:
            state["fired"] = True
            raise KeyboardInterrupt  # as if the user pressed Ctrl+C part-way through
        real_sleep(min(seconds, 0.001))

    monkeypatch.setattr(build.time, "sleep", sleep)
    slow = teacher.chat
    monkeypatch.setattr(teacher, "chat", lambda *a, **k: (real_sleep(0.002), slow(*a, **k))[1])
    with pytest.raises(KeyboardInterrupt):
        build.solve(cfg, teacher)
    done = len(read_jsonl(p["traces"]))
    assert 30 <= done < 200  # it stopped promptly instead of running to the end
    monkeypatch.setattr(build.time, "sleep", real_sleep)
    build.solve(cfg, teacher)
    from collections import Counter
    traces = read_jsonl(p["traces"])
    kept = [t["seed_id"] for t in traces if t["ok"]]
    assert len(kept) == len(set(kept)) >= 300                       # the resumed run reached its target
    assert max(Counter(t["seed_id"] for t in traces).values()) <= 2  # and no question went past its two tries


def test_half_written_last_line_is_ignored(tmp_path):
    f = tmp_path / "t.jsonl"
    f.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "b": "cut o')
    assert read_jsonl(f) == [{"a": 1}, {"a": 2}]
    f.write_text('{"a": 1}\n{"broken\n{"a": 3}\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(f)


def test_questions_are_written_in_parallel_without_duplicates(cfg):
    cfg["teacher"]["concurrency"] = 4
    teacher = ScriptedTeacher({})
    build.author(cfg, teacher)
    rows = read_jsonl(build.paths(cfg)["teacher_authored"])
    assert teacher.peak > 1                                  # requests really overlapped
    assert len(rows) >= 240 and len({r["id"] for r in rows}) == len(rows)
    assert len({r["generator"] for r in rows}) > 30 and {r["difficulty"] for r in rows} == {1, 2, 3, 4, 5}
    before = teacher.authored
    build.author(cfg, teacher)                               # already full: nothing more is requested
    assert teacher.authored == before


def test_sample_prints_kept_traces(cfg, capsys):
    build.sample(cfg)
    assert "no kept traces yet" in capsys.readouterr().out
    run(cfg)
    build.sample(cfg, 3)
    out = capsys.readouterr().out
    assert out.count("QUESTION") == 3 and "REASONING (27 words)" in out and "Final answer:" in out


def test_a_fault_in_checking_one_reply_is_logged_and_the_run_carries_on(cfg, monkeypatch):
    build.prepare(cfg)
    p = build.paths(cfg)
    sheet = {s["question"]: s["ground_truth"] for s in read_jsonl(p["procedural"])}
    cfg["dataset"]["mix"] = {"procedural": 1.0, "teacher_authored": 0.0, "public": 0.0}
    real, n = build.judge, {"calls": 0}

    def flaky(*a, **k):
        n["calls"] += 1
        if n["calls"] % 9 == 0:
            raise ValueError("Invalid literal for Fraction: '2.5 cups'")
        return real(*a, **k)

    monkeypatch.setattr(build, "judge", flaky)
    stats = build.solve(cfg, ScriptedTeacher(sheet))        # must finish, not hang or raise
    traces = read_jsonl(p["traces"])
    assert stats["reasons"]["check_error_ValueError"] > 5
    # every reply was logged, and the faults cost only the replies they hit
    assert len(traces) == stats["done"] and sum(t["ok"] for t in traces) == stats["kept"] > 250


def test_a_dead_worker_stops_the_others_and_reports_its_error():
    stop, started = threading.Event(), threading.Event()
    turn = iter(range(100))

    def worker():
        if next(turn) == 0:
            started.wait(2)
            raise OSError("disk full")
        started.set()
        while not stop.is_set():   # like a solver thread waiting on a question that will never come back
            time.sleep(0.01)

    t0 = time.time()
    with pytest.raises(OSError, match="disk full"):
        build.run_workers(worker, 4, stop, "test")
    assert stop.is_set() and time.time() - t0 < 5


def test_units_in_a_teacher_written_key_no_longer_break_solving(cfg):
    q = cfg["quality"]
    seed = {"difficulty": 2, "answer_type": "number", "ground_truth": "2.5 cups", "meta": {}, "mode": "think"}
    assert build.judge(seed, THINK, "Scale the recipe up.\n\nFinal answer: 2.5", "stop", q) == (True, [], "2.5")
    assert build.judge(seed, THINK, "Scale the recipe up.\n\nFinal answer: 3", "stop", q)[1] == ["wrong_answer"]
