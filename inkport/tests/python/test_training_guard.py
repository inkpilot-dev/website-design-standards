"""Training that breaks must recover, and must never hand a broken model to the final exam.

Seen on a Mac test machine: the same settings that gave a model passing 95% of the hidden exam
gave, the next time, one passing 0% (the original: 58%). The run ended in a rejection that said
"the improvement is too small". These tests run the real loop (schedule, practice evaluation,
checkpoints, final exam) on a stand-in for the model that can be made to break on purpose.
"""

import hashlib
import json
import math
from pathlib import Path

import pytest

from engine import events, smoke
from engine.backends.base import Backend
from engine.contract import ProjectPaths, read_jsonl
from engine.hardware_profile import make_plan, select_tier
from engine.pipeline import train_and_gate as tg
from engine.pipeline.ingest_and_verify import run_ingest

GARBAGE = "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"


def known(question: str) -> float:
    """Where a question sits between 0 and 1: the stand-in answers those below its skill. (A real
    hash: the questions differ in a few characters, and a sum of bytes put them in clumps.)"""
    return int(hashlib.sha256(question.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


class Stand(Backend):
    """Answers from a table. Knows `base` of the answers before training and more after each
    step, unless a step is taken with a learning rate above `breaks_above`: then it only
    produces garbage, the way a model whose weights went bad does. Learning rates are counted
    as a share of the run's own (1.0 = the rate the plan starts from), so that these tests say
    the same thing whatever the default rate is."""

    name = "torch"
    answers: dict = {}
    general: dict = {}  # stock general questions -> their right answers (the original knows them all)
    base = 0.6
    breaks_above = 9.0  # a share of the plan's learning rate; 9.0 = never
    forgets_above = 9.0  # a share above which it learns the task but loses the general questions...
    rehearsal_cures = 1  # ...unless a step also holds this many micro-batches of nothing but general questions
    broken_drift = 0.0  # what the measure of forgetting reads once broken (may be nan)
    always_drifts = 0.0  # drift added per step even when healthy
    log: list = []
    scales: list = []  # the weight each micro-batch was given

    def __init__(self, plan):
        super().__init__(plan)
        self.w, self.broken, self.forgot, self.adapter, self.steps = 0.0, False, False, False, 0
        self.rehearsal = 0  # micro-batches of general questions only, in the step being built

    # -- loading
    def load(self, adapter_dir=None, trainable=False):
        if adapter_dir is not None:
            self.load_checkpoint(Path(adapter_dir))
            self.adapter = True

    @property
    def has_adapter(self):
        return self.adapter

    # -- text
    def render(self, messages, add_generation_prompt=True):
        return "\n".join(m["content"] for m in messages)

    def _answer(self, conv, use_adapter):
        question = conv[-1]["content"]
        trained = use_adapter and self.adapter
        if question in self.general:
            return "no idea" if trained and (self.broken or self.forgot) else self.general[question]
        right = self.answers.get(question)
        if right is None:
            return "OK"  # free-form general prompts: the same answer before and after
        if trained and self.broken:
            return GARBAGE
        skill = self.base + (min(0.35, self.w) if trained else 0.0)
        rank = known(question)
        return right if rank < skill else json.dumps({"category": "none", "amount": 1})

    def generate(self, convs, max_new_tokens, temperature=0.0, top_p=1.0, n=1, seed=0, use_adapter=True):
        return [[self._answer(c, use_adapter)] * n for c in convs]

    # -- training
    def prepare_training(self):
        self.adapter = True
        self.reset_optimizer()

    def reset_optimizer(self):
        Stand.log.append("new optimizer")

    def train_microbatch(self, examples, scale):
        Stand.scales.append(scale)
        if all("check" in ex for ex in examples):
            self.rehearsal += 1
        return 1.0, 10 * len(examples)

    def optimizer_step(self, lr, max_grad_norm):
        self.steps += 1
        share = lr / float(self.plan["sft"]["lr"])
        if share > self.breaks_above:
            self.broken = True
        if self.rehearsal >= self.rehearsal_cures:
            self.forgot = False
        elif share > self.forgets_above:
            self.forgot = True
        self.rehearsal = 0
        self.w += 0.03 * share
        return 0.5

    def nll(self, examples):
        return 1.0 + self.always_drifts * self.steps + (self.broken_drift if self.broken else 0.0)

    def save_checkpoint(self, d):
        d.mkdir(parents=True, exist_ok=True)
        (d / "stand.json").write_text(json.dumps({"w": self.w, "broken": self.broken, "forgot": self.forgot, "steps": self.steps}))

    def load_checkpoint(self, d):
        state = json.loads((d / "stand.json").read_text())
        self.w, self.broken, self.forgot, self.steps = state["w"], state["broken"], state["forgot"], state["steps"]

    def save_adapter(self, d):
        self.save_checkpoint(d)
        self.write_adapter_meta(d)


@pytest.fixture()
def trained(project, tmp_path, monkeypatch):
    """Returns run(**settings) -> (result, log lines, report): a whole training run on the samples."""
    data = tmp_path / "data"
    smoke.write_smoke_data(data)
    # The same split every time (a project normally draws its own): these tests are about the
    # loop, and their numbers should not depend on which questions happen to land in the exam.
    meta = json.loads((project / "project.json").read_text())
    (project / "project.json").write_text(json.dumps({**meta, "split_salt": "7" * 32}))
    summary = run_ingest(project, [str(data / "expenses.csv"), str(data / "policy.md")], ["amount must be positive"])
    assert summary["ready"], summary
    answers = {}
    for split in ("train", "dev", "test"):
        for item in read_jsonl(ProjectPaths(project).split(split)):
            if item.get("reference"):
                answers[item["messages"][-1]["content"]] = item["reference"]
    assert len(answers) > 100
    lines: list = []
    monkeypatch.setattr(events, "emit", lambda event_type, **f: lines.append((event_type, f)) or {"type": event_type, **f})
    monkeypatch.setattr(tg, "make_backend", lambda plan, inference_only=False: Stand(plan))
    monkeypatch.delenv("INKPORT_LLAMA_CPP", raising=False)

    from engine.suites import general

    stock = {it["messages"][-1]["content"]: it["reference"] for it in general.replay_prompts(132) + general.suite(60) if it.get("reference")}

    def run(plan=None, **settings):
        Stand.answers, Stand.general, Stand.log, Stand.scales = answers, stock, [], []
        Stand.base, Stand.breaks_above, Stand.forgets_above, Stand.rehearsal_cures, Stand.broken_drift, Stand.always_drifts = 0.6, 9.0, 9.0, 1, 0.0, 0.0
        for key, value in settings.items():
            setattr(Stand, key, value)
        del lines[:]
        report = {"gpus": [], "ram_gb": 16, "apple_silicon": False}
        report["recommendation"] = select_tier(report)
        meta = json.loads((project / "project.json").read_text())
        overrides = {"sft": {"max_steps": 40, "eval_every": 20, "checkpoint_every": 20, "micro_batch": 4, "grad_accum": 1, **((plan or {}).get("sft") or {})}, "stage2": {"mode": "none"}, "gate": {"general_suite_n": 12}}
        result = tg.run_train(project, make_plan(meta, report, preset="fast", overrides=overrides))
        text = [f.get("message") or f.get("detail") or "" for kind, f in lines if kind in ("log", "stage")]
        return result, text, json.loads(Path(result["report_path"]).read_text())

    return run


def _has(text, needle):
    return any(needle in line for line in text)


def test_a_healthy_run_is_left_alone(trained):
    result, text, report = trained()
    assert result["decision"] == "PROMOTE", result
    assert not _has(text, "went wrong")
    assert report["training"]["stopped"] == "completed" and report["training"]["went_wrong"] == 0
    assert report["checks"]["training"] is True


def test_a_run_that_breaks_starts_again_more_gently_and_succeeds(trained):
    # The full learning rate breaks this model; half of it does not.
    result, text, report = trained(breaks_above=0.75)
    assert _has(text, "Training went wrong at step 20"), text
    assert _has(text, "Starting this stage again, more gently")
    assert result["decision"] == "PROMOTE", (result, text)
    assert report["training"]["went_wrong"] == 1 and report["training"]["lr_mult"] == 0.5
    assert report["test"]["candidate_pass_rate"] > report["test"]["baseline_pass_rate"]
    assert "new optimizer" in Stand.log[1:], "the restart must not reuse the optimizer of the broken attempt"


def test_broken_numbers_count_as_broken(trained):
    # Garbage answers and a measure of forgetting that is not a number: the practice score
    # alone would not catch this when the original passes few practice questions.
    result, text, _ = trained(breaks_above=0.75, broken_drift=math.nan, base=0.1)
    assert _has(text, "its numbers stopped being numbers"), text
    assert result["decision"] == "PROMOTE", (result, text)


def test_forgetting_at_the_first_check_no_longer_ships_the_drifted_model(trained):
    # Drift over the limit at step 20 used to end the stage and offer that very model. (The drift
    # is the measure of forgetting where general questions are not asked directly.)
    result, text, report = trained(breaks_above=0.75, broken_drift=0.4, plan={"sft": {"general_guard": False}})
    assert _has(text, "it is forgetting general skills"), text
    assert result["decision"] == "PROMOTE", (result, text)
    assert report["training"]["went_wrong"] == 1


def test_a_run_that_never_holds_ends_with_a_plain_verdict_and_saves_nothing(trained):
    result, text, report = trained(breaks_above=1e-9)
    assert sum("Training went wrong" in line for line in text) == 3, text
    assert result["decision"] == "REJECT" and result["active_version"] is None
    assert report["training"]["stopped"] == "unstable" and report["checks"]["training"] is False
    assert "did not hold steady" in result["explanation"] and "no trained model" in result["explanation"]
    assert "improvement is too small" not in result["explanation"]
    # What went to the final exam was the model as it was before training, not a broken one.
    assert report["test"]["candidate_pass_rate"] == report["test"]["baseline_pass_rate"]


def test_breaking_later_goes_back_to_the_best_version_so_far(trained):
    # Healthy until the second check, then forgetting: the best version (step 20) is what counts.
    result, text, report = trained(always_drifts=0.005, plan={"sft": {"general_guard": False}})  # 0.1 at step 20 (fine), 0.2 at step 40 (over 0.15)
    assert _has(text, "Training went wrong at step 40"), text
    assert _has(text, "Went back to the best version so far")
    assert result["decision"] == "PROMOTE", (result, text)
    assert report["training"]["went_wrong"] == 1


def test_where_general_questions_are_asked_directly_the_drift_reading_decides_nothing(trained):
    """Seen on the real model: a refinement round read a drift of +0.148 (the limit: 0.15) while
    the model answered 50 of the 60 watched general questions, two more than before the round. One
    more hundredth and a healthy round would have been thrown away as "forgetting general skills"."""
    result, text, report = trained(always_drifts=0.005)
    assert _has(text, "drift +0.200") and _has(text, "general questions 60/60"), text
    assert not _has(text, "Training went wrong")
    assert result["decision"] == "PROMOTE" and report["training"]["went_wrong"] == 0


def test_a_model_that_ends_up_worse_is_told_so(trained, monkeypatch):
    # Not broken, just worse: a model that unlearned what the original knew.
    class Worse(Stand):
        def _answer(self, conv, use_adapter):
            if use_adapter and self.adapter and self.w > 0.05 and conv[-1]["content"] in self.answers:
                rank = known(conv[-1]["content"])
                return self.answers[conv[-1]["content"]] if rank < 0.35 else json.dumps({"category": "none", "amount": 1})
            return super()._answer(conv, use_adapter)

    monkeypatch.setattr(tg, "make_backend", lambda plan, inference_only=False: Worse(plan))
    result, text, _ = trained()
    assert result["decision"] == "REJECT"
    assert "it did worse than the base on the hidden questions" in result["explanation"], result["explanation"]


def test_losing_general_questions_brings_more_rehearsal_and_keeps_the_progress(trained):
    # This model learns the task and, at any learning rate, loses the general questions unless
    # each step also rehearses some. Found at the first check, not at the exam, and nothing is
    # thrown away: the step counter goes on, rehearsal is added, the next version keeps them.
    result, text, report = trained(forgets_above=1e-9)
    assert _has(text, "general questions 0/60 (the original: 60)"), text
    assert _has(text, "It no longer answers 60 of the 60 general questions"), text
    assert not _has(text, "Training went wrong") and not _has(text, "Starting this stage again")
    assert _has(text, "sft step 40/40") and _has(text, "general questions 60/60"), text
    assert result["decision"] == "PROMOTE", (result, text)
    assert report["general"]["drop_pts"] == 0
    assert report["training"]["rehearsed_more"] == 1 and report["training"]["kept_general"] is True
    # Each step is still one average: with one rehearsal micro-batch added to one, each weighs a half.
    assert set(Stand.scales[:20]) == {1.0} and set(Stand.scales[-20:]) == {0.5}, (Stand.scales[:3], Stand.scales[-3:])


def test_a_version_that_lost_them_is_never_chosen_while_one_that_kept_them_exists(trained):
    # Fine at the first check; from then on it learns faster but loses the general questions
    # whatever is rehearsed. The first version is the one that goes to the exam.
    class Greedy(Stand):
        def optimizer_step(self, lr, max_grad_norm):
            norm = super().optimizer_step(lr, max_grad_norm)
            if self.steps > 20:
                self.forgot = True
            return norm

    import engine.pipeline.train_and_gate as module

    original = module.make_backend
    module.make_backend = lambda plan, inference_only=False: Greedy(plan)
    try:
        result, text, report = trained()
    finally:
        module.make_backend = original
    assert _has(text, "It no longer answers 60 of the 60 general questions"), text
    assert report["training"]["kept_general"] is True and report["general"]["drop_pts"] == 0
    assert result["decision"] == "PROMOTE", (result, text)


def test_of_two_equally_good_versions_the_one_that_kept_more_general_questions_is_chosen(trained):
    """Seen on the real model: every check from step 20 on passed all practice questions, the
    first of them was kept, and that was the one that had lost most general questions."""
    from engine.suites import general

    slow_to_return = {it["messages"][0]["content"] for it in general.replay_prompts(132)[:2]}

    class Quick(Stand):  # knows the task after a few steps; two general questions come back late
        def optimizer_step(self, lr, max_grad_norm):
            norm = super().optimizer_step(lr, max_grad_norm)
            self.w = 1.0
            return norm

        def _answer(self, conv, use_adapter):
            if use_adapter and self.adapter and self.steps <= 20 and conv[-1]["content"] in slow_to_return:
                return "no idea"
            return super()._answer(conv, use_adapter)

    import engine.pipeline.train_and_gate as module

    original = module.make_backend
    module.make_backend = lambda plan, inference_only=False: Quick(plan)
    try:
        result, text, report = trained()
    finally:
        module.make_backend = original
    assert _has(text, "general questions 58/60") and _has(text, "general questions 60/60"), text
    assert not _has(text, "It no longer answers")  # two of sixty is within the limit
    assert report["training"]["general_answered"] == 60 and report["training"]["general_watched"] == 60
    assert result["decision"] == "PROMOTE", (result, text)


def test_a_dip_that_chance_can_explain_is_called_that(trained):
    # One of the exam's twelve general questions lost: 8 points, more than the 5 allowed, but one
    # question proves nothing. The model is promoted, and the verdict does not call that "within
    # the allowed limit".
    from engine.suites import general

    one = general.suite(12)[0]["messages"][0]["content"]

    class Slip(Stand):
        def _answer(self, conv, use_adapter):
            if use_adapter and self.adapter and conv[-1]["content"] == one:
                return "no idea"
            return super()._answer(conv, use_adapter)

    import engine.pipeline.train_and_gate as module

    original = module.make_backend
    module.make_backend = lambda plan, inference_only=False: Slip(plan)
    try:
        result, text, report = trained()
    finally:
        module.make_backend = original
    assert result["decision"] == "PROMOTE", result
    assert "General skills dipped 8.3 points on 12 questions, which chance alone can do" in result["explanation"], result["explanation"]
    assert "within the allowed limit" not in result["explanation"]


def test_without_that_watch_the_same_run_is_refused_only_at_the_exam(trained):
    result, text, report = trained(forgets_above=1e-9, plan={"sft": {"general_guard": False}})
    assert not _has(text, "It no longer answers") and not _has(text, "general questions 0/")
    assert result["decision"] == "REJECT" and "general skills dropped" in result["explanation"], result
    assert _has(text, "General question no longer answered correctly"), text


def test_a_model_that_cannot_keep_them_still_gets_its_exam_and_an_honest_verdict(trained):
    # Rehearsal does not help this one. It is not "broken": the version that learned the task best
    # sits the exam, and the verdict names the real reason.
    result, text, report = trained(forgets_above=1e-9, rehearsal_cures=99)
    assert _has(text, "No version of this stage kept its general skills"), text
    assert report["training"]["stopped"] == "completed" and report["training"]["kept_general"] is False
    assert report["training"]["rehearsed_more"] == 2
    assert result["decision"] == "REJECT" and "general skills dropped" in result["explanation"], result["explanation"]
    assert "did not hold steady" not in result["explanation"]


def test_a_model_that_was_already_good_is_told_that_and_not_that_training_failed(trained):
    """Seen with the bundled model on the documents and SQL samples: the original passes 29 to 32
    of the 32 to 35 hidden questions, the trained one all of them, and the verdict said "the
    improvement is too small or not statistically reliable yet", as if more training could help."""
    result, text, report = trained(base=0.97)
    missed = round(report["test"]["n"] * (1 - report["test"]["baseline_pass_rate"]))
    assert 0 < missed < 6, report["test"]
    assert report["test"]["candidate_pass_rate"] == 1.0
    assert result["decision"] == "REJECT"
    assert f"the base already answers {report['test']['n'] - missed} of the {report['test']['n']} hidden questions correctly" in result["explanation"], result["explanation"]
    assert "too few for an improvement to be told from chance" in result["explanation"]
    assert "add rules that say what is missing" in result["explanation"]
    assert "improvement is too small" not in result["explanation"]


def test_when_there_is_no_room_to_improve():
    # Fewer than 100 hidden questions: six won back with none lost is the least that counts.
    assert tg.no_room_to_improve(32, 35) and tg.no_room_to_improve(30, 35)
    assert not tg.no_room_to_improve(29, 35) and not tg.no_room_to_improve(0, 39)
    assert tg.no_room_to_improve(35, 35)
    # 100 or more: four are enough for a range above zero, but they must also be two points.
    assert tg.no_room_to_improve(117, 120, verified=True) and not tg.no_room_to_improve(116, 120, verified=True)
    assert tg.no_room_to_improve(396, 400, verified=True)  # four of 400 is one point


def test_few_examples_get_more_passes_instead_of_a_higher_learning_rate():
    cfg = {"min_steps": 60, "small_set_passes": 12, "max_steps": None}
    assert tg.planned_steps(213, 8, 3, cfg) == (27, 81)  # the expenses sample: three passes, as before
    assert tg.planned_steps(87, 8, 3, cfg) == (11, 60)  # the customer-replies sample: 33 steps before
    assert tg.planned_steps(10, 8, 3, cfg) == (2, 24)  # a handful of examples: a dozen passes, not thirty
    assert tg.planned_steps(87, 8, 1, cfg, first_stage=False) == (11, 11)  # refinement rounds stay short
    assert tg.planned_steps(87, 8, 3, {**cfg, "max_steps": 40}) == (11, 40)
    assert tg.planned_steps(87, 8, 3, {}) == (11, 33)  # a plan written before this existed


def test_the_recipe_a_user_gets(project):
    report = {"gpus": [], "ram_gb": 16, "apple_silicon": False}
    report["recommendation"] = select_tier(report)
    meta = json.loads((project / "project.json").read_text())
    sft = make_plan(meta, report, preset="fast")["sft"]
    # Twice this rate cost the bundled model most of its general skills by the first practice check.
    assert sft["lr"] == 1e-4 and sft["min_steps"] == 60 and sft["small_set_passes"] == 12
    assert sft.get("general_guard", True) is True
    tiny = make_plan(meta, report, preset="smoke")["sft"]
    assert tiny["min_steps"] == 0 and tiny["general_guard"] is False


def test_how_many_general_questions_may_be_lost():
    assert [tg.general_limit(n) for n in (8, 16, 24, 40, 60, 100)] == [2, 2, 2, 2, 3, 5]


def test_the_signs_of_a_broken_model():
    ok = {"n": 32, "passed": 20, "results": []}
    assert tg.gone_wrong([True] * 18 + [False] * 14, None, ok, 0.05, 0.15) is None
    assert tg.gone_wrong([True] * 18 + [False] * 14, None, {"n": 32, "passed": 9, "results": []}, 0.05, 0.15) is None  # worse, not broken
    assert "practice questions" in tg.gone_wrong([True] * 18 + [False] * 14, None, {"n": 32, "passed": 4, "results": []}, 0.05, 0.15)
    assert tg.gone_wrong([True] * 6 + [False] * 26, None, {"n": 32, "passed": 0, "results": []}, 0.05, 0.15) is None  # too few to tell
    assert "forgetting" in tg.gone_wrong([], None, ok, 0.2, 0.15)
    assert "numbers" in tg.gone_wrong([], None, ok, math.nan, 0.15)
    malformed = {"n": 4, "passed": 0, "results": [{"verdicts": [{"kind": "json_valid", "pass": i < 1}]} for i in range(4)]}
    assert "well-formed" in tg.gone_wrong([], 1.0, malformed, 0.0, 0.15)
    assert tg.gone_wrong([], 0.5, malformed, 0.0, 0.15) is None  # the original was no better


def test_what_the_exam_asks_is_never_rehearsed_and_nothing_is_asked_twice():
    """The general questions mixed into training, the ones watched while it runs and the ones of
    the final exam are three separate sets (35 of 132 rehearsal prompts used to be exam questions)."""
    from engine.suites import general

    rehearsal = [it["messages"][0]["content"] for it in general.replay_prompts(132)]
    exam = {it["messages"][0]["content"] for it in general.suite(240)}
    assert len(set(rehearsal)) == len(rehearsal) == 132
    assert not set(rehearsal) & exam
    watched, pool = rehearsal[: tg.WATCHED], rehearsal[tg.WATCHED :]
    assert not set(watched) & set(pool)
    assert all(it["check"] for it in general.replay_prompts(132)[: tg.WATCHED]), "every watched question can be judged"
    # Still distinct when more are asked for than there are words.
    many = [it["messages"][0]["content"] for it in general.replay_prompts(400)]
    assert len(set(many)) == 400 and not set(many) & exam


def test_the_exam_itself_did_not_change():
    """Results stored for the original model are compared with new ones question by question."""
    from engine.suites import general

    first = [it["messages"][0]["content"] for it in general.suite(60)[:6]]
    assert first == [
        "What is 22 + 11? Reply with only the number.",
        "Reply with only the word: north",
        "Write the word 'harbor' in uppercase letters only.",
        'Return a JSON object with the key "color" set to "river". Reply with JSON only.',
        "How many letters are in the word 'meadow'? Reply with only the number.",
        "Which is larger, 73 or 17? Reply with only the number.",
    ], first
