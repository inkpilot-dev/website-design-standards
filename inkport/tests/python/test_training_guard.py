"""Training that breaks must recover, and must never hand a broken model to the final exam.

Seen on a Mac test machine: the same settings that gave a model passing 95% of the hidden exam
gave, the next time, one passing 0% (the original: 58%). The run ended in a rejection that said
"the improvement is too small". These tests run the real loop (schedule, practice evaluation,
checkpoints, final exam) on a stand-in for the model that can be made to break on purpose.
"""

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


class Stand(Backend):
    """Answers from a table. Knows `base` of the answers before training and more after each
    step, unless a step is taken with a learning rate above `breaks_above`: then it only
    produces garbage, the way a model whose weights went bad does."""

    name = "torch"
    answers: dict = {}
    base = 0.6
    breaks_above = 1.0  # a learning rate; 1.0 = never
    broken_drift = 0.0  # what the measure of forgetting reads once broken (may be nan)
    always_drifts = 0.0  # drift added per step even when healthy
    log: list = []

    def __init__(self, plan):
        super().__init__(plan)
        self.w, self.broken, self.adapter, self.steps = 0.0, False, False, 0

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
        right = self.answers.get(question)
        if right is None:
            return "OK"  # general prompts: the same answer before and after
        if use_adapter and self.adapter and self.broken:
            return GARBAGE
        skill = self.base + (min(0.35, self.w) if (use_adapter and self.adapter) else 0.0)
        rank = (sum(question.encode()) % 100) / 100.0
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
        return 1.0, 10 * len(examples)

    def optimizer_step(self, lr, max_grad_norm):
        self.steps += 1
        if lr > self.breaks_above:
            self.broken = True
        self.w += 60 * lr
        return 0.5

    def nll(self, examples):
        return 1.0 + self.always_drifts * self.steps + (self.broken_drift if self.broken else 0.0)

    def save_checkpoint(self, d):
        d.mkdir(parents=True, exist_ok=True)
        (d / "stand.json").write_text(json.dumps({"w": self.w, "broken": self.broken, "steps": self.steps}))

    def load_checkpoint(self, d):
        state = json.loads((d / "stand.json").read_text())
        self.w, self.broken, self.steps = state["w"], state["broken"], state["steps"]

    def save_adapter(self, d):
        self.save_checkpoint(d)
        self.write_adapter_meta(d)


@pytest.fixture()
def trained(project, tmp_path, monkeypatch):
    """Returns run(**settings) -> (result, log lines, report): a whole training run on the samples."""
    data = tmp_path / "data"
    smoke.write_smoke_data(data)
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

    def run(**settings):
        Stand.answers, Stand.log = answers, []
        Stand.base, Stand.breaks_above, Stand.broken_drift, Stand.always_drifts = 0.6, 1.0, 0.0, 0.0
        for key, value in settings.items():
            setattr(Stand, key, value)
        del lines[:]
        report = {"gpus": [], "ram_gb": 16, "apple_silicon": False}
        report["recommendation"] = select_tier(report)
        meta = json.loads((project / "project.json").read_text())
        plan = make_plan(meta, report, preset="fast", overrides={"sft": {"max_steps": 40, "eval_every": 20, "checkpoint_every": 20, "micro_batch": 4, "grad_accum": 1}, "stage2": {"mode": "none"}, "gate": {"general_suite_n": 8}})
        result = tg.run_train(project, plan)
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
    # The full learning rate (2e-4) breaks this model; half of it does not.
    result, text, report = trained(breaks_above=1.5e-4)
    assert _has(text, "Training went wrong at step 20"), text
    assert _has(text, "Starting this stage again, more gently")
    assert result["decision"] == "PROMOTE", (result, text)
    assert report["training"]["went_wrong"] == 1 and report["training"]["lr_mult"] == 0.5
    assert report["test"]["candidate_pass_rate"] > report["test"]["baseline_pass_rate"]
    assert "new optimizer" in Stand.log[1:], "the restart must not reuse the optimizer of the broken attempt"


def test_broken_numbers_count_as_broken(trained):
    # Garbage answers and a measure of forgetting that is not a number: the practice score
    # alone would not catch this when the original passes few practice questions.
    result, text, _ = trained(breaks_above=1.5e-4, broken_drift=math.nan, base=0.1)
    assert _has(text, "its numbers stopped being numbers"), text
    assert result["decision"] == "PROMOTE", (result, text)


def test_forgetting_at_the_first_check_no_longer_ships_the_drifted_model(trained):
    # Drift over the limit at step 20 used to end the stage and offer that very model.
    result, text, report = trained(breaks_above=1.5e-4, broken_drift=0.4)
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
    result, text, report = trained(always_drifts=0.005)  # 0.1 at step 20 (fine), 0.2 at step 40 (over 0.15)
    assert _has(text, "Training went wrong at step 40"), text
    assert _has(text, "Went back to the best version so far")
    assert result["decision"] == "PROMOTE", (result, text)
    assert report["training"]["went_wrong"] == 1


def test_a_model_that_ends_up_worse_is_told_so(trained, monkeypatch):
    # Not broken, just worse: a model that unlearned what the original knew.
    class Worse(Stand):
        def _answer(self, conv, use_adapter):
            if use_adapter and self.adapter and self.w > 0.05 and conv[-1]["content"] in self.answers:
                rank = (sum(conv[-1]["content"].encode()) % 100) / 100.0
                return self.answers[conv[-1]["content"]] if rank < 0.35 else json.dumps({"category": "none", "amount": 1})
            return super()._answer(conv, use_adapter)

    monkeypatch.setattr(tg, "make_backend", lambda plan, inference_only=False: Worse(plan))
    result, text, _ = trained()
    assert result["decision"] == "REJECT"
    assert "it did worse than the base on the hidden questions" in result["explanation"], result["explanation"]


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
