"""Two-stage training with crash-resume, then the Before-vs-After gate with automatic rollback.

Stages (each recorded in runs/<run_id>/state.json so `engine train --resume` skips finished work):

  verify_data -> baseline -> replay -> sft -> stage2 (rft | grpo | none) -> gate -> promote

* verify_data: refuses to train if any train/dev item or group also appears in the hidden test
  set, or if the test set no longer matches the Merkle root written at ingestion.
* baseline: the untouched base model (and the currently active version, if any) answer every
  hidden test question; results are cached per (model, test set).
* replay: the base model answers general prompts; those answers are mixed into SFT and a held
  out slice measures drift (mean token NLL increase) as the forgetting guard.
* sft: QLoRA/LoRA/DoRA with warmup + cosine LR, gradient accumulation, periodic dev evaluation,
  best-checkpoint selection, early stopping, the forgetting guard, and resumable checkpoints.
* stage2: outcome-filtered refinement (sample k answers, keep only verifier-passing ones,
  train on them) or TRL GRPO with the verifiers as the reward. Kept only if dev improves.
* gate: candidate vs incumbent on the hidden test set (Wilson CIs, exact McNemar, paired
  bootstrap), general-skills regression, format regression and audit checks. Only PROMOTE
  changes active.json; anything else leaves the current version in place (automatic rollback).
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from engine import events
from engine.backends.base import Backend, make_backend
from engine.contract import ProjectPaths, active_version, atomic_write_json, load_project, read_jsonl, save_project, sha256_dir, write_jsonl
from engine.pipeline.ingest_and_verify import merkle_root
from engine.stats import discordant, mcnemar_exact, paired_bootstrap, wilson
from engine.suites import general
from engine.verify.core import Context, load_verifier_set, score_batch

OVERALL = {"baseline": (2, 18), "replay": (18, 22), "sft": (22, 70), "stage2": (70, 84), "gate": (84, 98), "promote": (98, 100)}
WATCHED = 60  # general questions held back from training and asked at every practice evaluation


class LeakageError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------
# State
# ------------------------------------------------------------------------------------------


class RunState:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "state.json"
        self.data: dict[str, Any] = json.loads(self.path.read_text()) if self.path.is_file() else {"completed": [], "sft": {}}

    def done(self, stage: str) -> bool:
        return stage in self.data["completed"]

    def complete(self, stage: str, **info: Any) -> None:
        if stage not in self.data["completed"]:
            self.data["completed"].append(stage)
        self.data.setdefault("info", {})[stage] = info
        self.save()

    def save(self) -> None:
        atomic_write_json(self.path, self.data)


# ------------------------------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------------------------------


def lr_at(step: int, total: int, peak: float, warmup_frac: float, min_ratio: float) -> float:
    warm = max(1, min(max(10, int(warmup_frac * total)), max(1, total // 2)))
    if step < warm:
        return peak * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def evaluate(backend: Backend, items: list[dict], vset: dict, ctx: Context, max_new_tokens: int, use_adapter: bool = True, progress: Optional[events.Progress] = None, offset: int = 0) -> dict:
    """Greedy generation + verifiers. Returns {"outputs", "results", "pass", "n", "passed"}."""
    outputs: list[str] = []
    bs = 16
    for i in range(0, len(items), bs):
        chunk = items[i : i + bs]
        outs = backend.generate([it["messages"] for it in chunk], max_new_tokens=max_new_tokens, temperature=0.0, use_adapter=use_adapter)
        outputs += [o[0] for o in outs]
        if progress:
            progress.update(offset + i + len(chunk))
    results = score_batch(items, outputs, vset, ctx)
    passes = [r["pass"] for r in results]
    return {"outputs": outputs, "results": results, "pass": passes, "n": len(items), "passed": sum(passes)}


def eval_general(backend: Backend, n: int, use_adapter: bool = True) -> dict:
    items = general.suite(n)
    outs = backend.generate([it["messages"] for it in items], max_new_tokens=24, temperature=0.0, use_adapter=use_adapter)
    passes = [general.check(it, o[0]) for it, o in zip(items, outs)]
    return {"pass": passes, "n": len(items), "passed": sum(passes), "outputs": [o[0] for o in outs]}


def emit_eval(split: str, model: str, res: dict) -> None:
    lo, hi = wilson(res["passed"], res["n"])
    events.emit("eval", split=split, model=model, n=res["n"], passed=res["passed"], pass_rate=(res["passed"] / res["n"]) if res["n"] else 0.0, ci95=[round(lo, 4), round(hi, 4)])


def format_rate(res: dict) -> Optional[float]:
    vals = []
    for r in res["results"]:
        for v in r["verdicts"]:
            if v["kind"] == "json_valid" and v["pass"] is not None:
                vals.append(bool(v["pass"]))
    return (sum(vals) / len(vals)) if vals else None


def format_pass(res: dict) -> list[Optional[bool]]:
    """Per test item: did every JSON-validity check pass (None when the item has no such check)."""
    out: list[Optional[bool]] = []
    for r in res["results"]:
        vals = [bool(v["pass"]) for v in r["verdicts"] if v["kind"] == "json_valid" and v["pass"] is not None]
        out.append(all(vals) if vals else None)
    return out


def format_check(f_ref: Optional[float], f_cand: Optional[float], ref_items: Optional[list], cand_items: list, task_gain: Optional[int] = None) -> bool:
    """False only when the candidate formats answers worse than the reference, beyond noise, and
    that loss is not clearly outweighed by what it gained on the task.

    A drop of more than 2 points must be statistically visible on the paired items (exact McNemar,
    p <= 0.10), like the general-skills check: on ~90 questions two malformed answers are already
    2.3 points.

    And a malformed answer already counts as a wrong answer in the task score. A model that gets
    35 more of 96 questions right and leaves 5 answers malformed is simply the better model; that
    case (95% against 58%) was once refused "because it formats answers correctly less often".
    So the loss in well-formed answers only blocks a model whose net gain in correct answers
    (``task_gain``) is less than twice that loss.
    """
    if f_ref is None or f_cand is None or f_cand >= f_ref - 0.02:
        return True
    if not ref_items or len(ref_items) != len(cand_items):
        return False  # no per-item data (older cached baseline): keep the plain threshold
    pairs = [(bool(x), bool(y)) for x, y in zip(ref_items, cand_items) if x is not None and y is not None]
    gained, lost = discordant([x for x, _ in pairs], [y for _, y in pairs])
    if mcnemar_exact(gained, lost) > 0.10:
        return True
    return task_gain is not None and task_gain >= 2 * (lost - gained)


def general_limit(n: int) -> int:
    """From how many of ``n`` general questions lost (net) a version is no longer one to keep.

    The final exam refuses a model that loses more than 5 points on 60 general questions when that
    loss is beyond chance, which works out at 5 questions of 60. The exam asks other questions
    than the ones watched here, so a version that lost 4 of these could well lose 5 of those: the
    line is drawn at 3 of 60, which leaves room for the luck of the draw. Never below 2: one
    question changing hands is noise.
    """
    return max(2, math.ceil(n / 20))  # 3 of the 60 that are watched


def gone_wrong(base_pass: list, base_format: Optional[float], res: dict, drift: float, drift_limit: float) -> Optional[str]:
    """Why the model being trained must not be kept as it is now (None: nothing wrong).

    Seen on a real machine: a run whose result passed 0% of the hidden exam against 58% for the
    original, where the same settings had given 95% the time before. A run like that is not "a bit
    worse", it has broken, and the signs are visible at the first practice evaluation:

    * the measure of forgetting is not a number, or is over its limit;
    * it passes a quarter or less of the practice questions the original passes (and the original
      passes at least 8, so this is not noise);
    * at most half of its answers are well-formed where the original's nearly all were.

    A model that is merely slow to improve trips none of these. (Losing general questions is not
    one of them either: that is a cost of learning, handled in the loop by rehearsing more.)
    """
    if not math.isfinite(drift):
        return "its numbers stopped being numbers"
    if drift > drift_limit:
        return f"it is forgetting general skills (drift {drift:.3f}, limit {drift_limit})"
    was = sum(1 for x in base_pass if x)
    if was >= 8 and 4 * res["passed"] <= was:
        return f"it passes {res['passed']} of the {res['n']} practice questions; the original passes {was}"
    now_format = format_rate(res)
    if base_format is not None and now_format is not None and base_format >= 0.8 and now_format <= 0.5:
        return f"only {now_format:.0%} of its practice answers are well-formed (the original: {base_format:.0%})"
    return None


def training_failed(info: Optional[dict]) -> bool:
    """The stage stopped because it kept going wrong, and never produced anything worth keeping."""
    return bool(info) and info.get("stopped") == "unstable" and float(info.get("best_dev", -1.0)) < 0


def audit_rate(res: dict) -> Optional[float]:
    """Among items that passed, the fraction that also pass the held-out audit verifiers."""
    vals = [r["audit_pass"] for r in res["results"] if r["pass"] and r["audit_pass"] is not None]
    return (sum(vals) / len(vals)) if vals else None


def to_example(item: dict) -> dict:
    return {"messages": item["messages"] + [{"role": "assistant", "content": item["reference"]}]}


def user_prompt(item: dict) -> str:
    return next(m["content"] for m in reversed(item["messages"]) if m["role"] == "user")


# ------------------------------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------------------------------


class Trainer:
    def __init__(self, root: Path, plan: dict):
        self.root = root
        self.plan = plan
        self.paths = ProjectPaths(root)
        self.project = load_project(root)
        self.run_dir = self.paths.run_dir(plan["run_id"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.run_dir / "plan.json", plan)
        self.state = RunState(self.run_dir)
        self.vset = load_verifier_set(root)
        self.ctx = Context(root=root)
        self.t0 = time.monotonic()
        self.backend: Optional[Backend] = None
        self.rng = random.Random(plan["seed"])
        self.train = [it for it in read_jsonl(self.paths.split("train")) if it.get("reference")]
        self.dev = [it for it in read_jsonl(self.paths.split("dev")) if it.get("reference") is not None]
        self.test = read_jsonl(self.paths.split("test"))
        self.gate_cfg = plan["gate"]
        self.max_new = int(self.gate_cfg["max_new_tokens"])
        self.training_info: Optional[dict] = None  # how the first training stage ended (run_train)

    # ---------------------------------------------------------------- utils ----------------
    def elapsed_min(self) -> float:
        return (time.monotonic() - self.t0) / 60.0

    def over_budget(self, frac: float = 1.0) -> bool:
        return self.elapsed_min() > frac * float(self.plan["budget_minutes"])

    def dev_subset(self) -> list[dict]:
        pool = self.dev if self.dev else self.train
        return pool[: int(self.plan.get("dev_eval_n", 32))]

    def load(self, adapter: Optional[Path] = None, trainable: bool = False) -> Backend:
        if self.backend is not None:
            self.backend.unload()
        events.log("Loading model" + (f" with adapter {adapter.name}" if adapter else ""), "debug")
        b = make_backend(self.plan)
        b.load(adapter, trainable=trainable)
        self.backend = b
        return b

    # ---------------------------------------------------------------- stages ---------------
    def verify_data(self) -> None:
        manifest = json.loads(self.paths.test_manifest.read_text())
        test_ids = {it["item_id"] for it in self.test}
        if merkle_root(sorted(test_ids)) != manifest["merkle_root"]:
            raise LeakageError("The hidden test set changed since it was locked. Re-run 'Build practice exam'.")
        test_groups = {it["group"] for it in self.test}
        for split, rows in (("train", self.train), ("dev", self.dev)):
            bad = [it["item_id"] for it in rows if it["item_id"] in test_ids or it["group"] in test_groups]
            if bad:
                raise LeakageError(f"{len(bad)} {split} item(s) overlap the hidden test set; refusing to train")
        if not self.train:
            raise ValueError("There are no training examples. Add more examples in “Data & rules” and build the practice exam again.")
        # Below this the gate can only answer "not enough data" (see gate()); say so before hours of training, not after.
        need = max(10, int(self.gate_cfg["min_test_items"]) // 2)
        if len(self.test) < need:
            raise ValueError(f"The hidden final exam has only {len(self.test)} question(s); at least {need} are needed to judge fairly. Add more examples in “Data & rules” and build the practice exam again.")
        if len(self.test) < int(self.gate_cfg["min_test_items"]):
            events.log(f"Only {len(self.test)} hidden test questions; the result will be marked preliminary.", "warn")
        events.log(f"Data verified: {len(self.train)} train, {len(self.dev)} dev, {len(self.test)} hidden test; zero overlap; test root {manifest['merkle_root'][:12]}", "info")

    def _cache_key(self, model_tag: str) -> Path:
        root = json.loads(self.paths.test_manifest.read_text())["merkle_root"]
        tag = f"{self.plan['base_model']['id']}|{self.plan['backend']}|{self.plan.get('quant')}|{model_tag}|{root}|{self.max_new}|{self.gate_cfg['general_suite_n']}"
        import hashlib  # noqa: WPS433

        return self.paths.cache / f"eval-{hashlib.sha256(tag.encode()).hexdigest()[:20]}.json"

    def baseline(self) -> dict:
        cache = self._cache_key("base")
        dev = self.dev_subset()
        if cache.is_file():
            data = json.loads(cache.read_text())
            if len(data.get("dev_outputs", [])) == len(dev):
                events.log("Using cached baseline results for this exact test set", "info")
                emit_eval("test", "base", data["test"])
                return data
        b = self.backend or self.load()
        o0, o1 = OVERALL["baseline"]
        total = len(self.test) + len(dev)
        prog = events.Progress("baseline", total, o0, o1)
        test = evaluate(b, self.test, self.vset, self.ctx, self.max_new, use_adapter=False, progress=prog)
        emit_eval("test", "base", test)
        dev_res = evaluate(b, dev, self.vset, self.ctx, self.max_new, use_adapter=False, progress=prog, offset=len(self.test))
        gen = eval_general(b, int(self.gate_cfg["general_suite_n"]), use_adapter=False)
        emit_eval("general", "base", gen)
        data = {
            "test": {k: test[k] for k in ("n", "passed", "pass")} | {"format_rate": format_rate(test), "format_pass": format_pass(test), "outputs": test["outputs"]},
            "dev_outputs": dev_res["outputs"],
            "dev_pass": dev_res["pass"],
            "general": gen,
        }
        atomic_write_json(cache, data)
        return data

    def replay(self) -> list[dict]:
        path = self.run_dir / "replay.jsonl"
        if path.is_file():
            return read_jsonl(path)
        b = self.backend or self.load()
        # 60 are held back from training and asked again while it runs (see sft); the rest are mixed
        # into the training examples.
        prompts = general.replay_prompts(int(self.plan["sft"].get("replay_prompts", 132)))
        outs = b.generate([p["messages"] for p in prompts], max_new_tokens=64, temperature=0.0, use_adapter=False)
        # "check": how an answer to this question is judged (None for the free-form ones), so that
        # the questions held back from training can be asked again while it runs.
        rows = [{"messages": p["messages"] + [{"role": "assistant", "content": o[0] or "OK"}], "check": p.get("check")} for p, o in zip(prompts, outs)]
        write_jsonl(path, rows)
        return rows

    # ---------------------------------------------------------------- SFT loop -------------
    def _dev_eval(self, baseline: dict, step: int) -> dict:
        b = self.backend
        dev = self.dev_subset()
        res = evaluate(b, dev, self.vset, self.ctx, self.max_new)
        for i, it in enumerate(dev[:4]):
            base_out = baseline["dev_outputs"][i] if i < len(baseline["dev_outputs"]) else ""
            base_pass = baseline["dev_pass"][i] if i < len(baseline["dev_pass"]) else None
            events.emit(
                "benchmark",
                source="training_dev",
                case_id=it["item_id"],
                prompt=user_prompt(it),
                reference=it.get("reference"),
                baseline={"output": base_out, "pass": base_pass, "score": None if base_pass is None else float(base_pass)},
                trained={"output": res["outputs"][i], "pass": res["results"][i]["pass"], "score": res["results"][i]["score"]},
                step=step,
                judge=None,
            )
        return res

    def sft(self, examples: list[dict], replay_rows: list[dict], baseline: dict, *, stage: str, lr: float, epochs: float, overall: tuple[float, float], resume: bool) -> dict:
        b = self.backend
        cfg = self.plan["sft"]
        # Held back from training: with enough of them, as many as the final exam asks (60), so that
        # losing them can be told from chance the way the exam does; from a shorter list, a quarter.
        n_held = WATCHED if len(replay_rows) >= WATCHED + 32 else max(4, len(replay_rows) // 4)
        held = replay_rows[:n_held]
        replay_train = replay_rows[n_held:]
        holdout = held[:16]  # the forgetting readings need no more than these
        rho = float(cfg["replay_frac"])
        n_replay = min(len(replay_train), int(round(rho / max(1e-9, 1 - rho) * len(examples)))) if rho > 0 else 0
        data = examples + replay_train[:n_replay]
        mb, ga = int(cfg["micro_batch"]), int(cfg["grad_accum"])
        per_step = mb * ga
        steps_per_epoch = max(1, math.ceil(len(data) / per_step))
        total = max(1, math.ceil(epochs * steps_per_epoch))
        if cfg.get("max_steps"):
            total = min(total, int(cfg["max_steps"]))
        eval_every = int(cfg["eval_every"]) or max(20, total // 10)
        ckpt_every = int(cfg["checkpoint_every"])
        st = self.state.data.setdefault(stage, {})
        ckpt = self.run_dir / f"ckpt-{stage}"
        best_dir = self.run_dir / f"best-{stage}"
        init_dir = self.run_dir / f"init-{stage}"
        # The best on the practice questions among the versions that lost too many general ones:
        # only used if no version at all kept them (the final exam then says so).
        spare_dir = self.run_dir / f"best-over-{stage}"
        step = 0
        best = st.get("best_dev", -1.0)
        spare = st.get("spare_dev", -1.0)
        # How many of the watched general questions the best (and the spare) version answers:
        # between two versions that pass as many practice questions, the one that kept more wins.
        best_general, spare_general = st.get("best_general", -1), st.get("spare_general", -1)
        patience = st.get("patience", 0)
        lr_mult = st.get("lr_mult", 1.0)
        guard_trips = st.get("guard_trips", 0)
        general_trips = st.get("general_trips", 0)
        extra = st.get("extra_rehearsal", 0)  # rehearsal micro-batches added to every step
        rehearsed = 0
        if resume and ckpt.is_dir() and st.get("step"):
            b.load_checkpoint(ckpt)
            step = int(st["step"])
            events.log(f"Resumed {stage} from step {step}", "info")
        else:
            # Where this stage starts from: the place to go back to if training breaks before
            # anything better exists.
            b.save_checkpoint(init_dir)
        dev = self.dev_subset()
        # The practice score and the answer format tell a broken model from a healthy one only if
        # the reference answers themselves pass those checks; a model must never be called broken
        # for learning what its examples say.
        refs = score_batch(dev, [str(it.get("reference") or "") for it in dev], self.vset, self.ctx)
        refs_pass = sum(1 for r in refs if r["pass"]) / max(1, len(refs))
        base_dev_pass = list(baseline.get("dev_pass") or [])[: len(dev)] if refs_pass >= 0.8 else []
        base_dev_format = None
        if refs_pass >= 0.8 and (format_rate({"results": refs}) or 0.0) >= 0.8 and len(baseline.get("dev_outputs") or []) == len(dev):
            base_dev_format = format_rate({"results": score_batch(dev, baseline["dev_outputs"], self.vset, self.ctx)})
        base_nll = st.get("base_nll")
        if base_nll is None:
            base_nll = b.nll(holdout)
            st["base_nll"] = base_nll
        # General questions held back from training, of the kind the final exam asks: asked again at
        # every practice evaluation, compared with how the original model answered them.
        watch = [row for row in held if row.get("check")]
        base_general = [general.check(row, row["messages"][-1]["content"]) for row in watch]
        watch_general = bool(cfg.get("general_guard", True)) and sum(base_general) >= 8
        # A second reading of forgetting, on the answers alone. The first also counts the tokens
        # that mark the end of an answer; every real-model run so far reported a "drift" of -0.8 to
        # -1.8 nats/token, which can only mean those end tokens cost the original model a lot and
        # are learned at once, hiding any real forgetting behind them. Logged next to the first so
        # that the limit can be set on real figures; the guard still uses the first.
        answers_only = [{**row, "open_end": True} for row in holdout]
        base_answers = st.get("base_nll_answers")
        if base_answers is None:
            base_answers = b.nll(answers_only)
            st["base_nll_answers"] = base_answers
            events.log(f"Before {stage}: {len(holdout)} answers to general questions, written by the original model, cost {base_answers:.3f} nats/token ({base_nll:.3f} with the end of each answer counted)", "info")
        prog = events.Progress(stage, total, overall[0], overall[1])
        prog.update(step, force=True)
        tokens, t_last, loss_ema = 0, time.monotonic(), None
        stopped = "completed"
        last_dev, last_dnll = st.get("last_dev"), None
        while step < total:
            epoch = step // steps_per_epoch
            order = list(range(len(data)))
            random.Random(self.plan["seed"] * 1000 + epoch + (17 if stage != "sft" else 0)).shuffle(order)
            pos = (step % steps_per_epoch) * per_step
            losses = []
            n_micro = ga + (extra if replay_train else 0)
            for k in range(ga):
                idx = order[pos + k * mb : pos + (k + 1) * mb]
                if not idx:
                    idx = order[: mb]
                loss, ntok = b.train_microbatch([data[i] for i in idx], 1.0 / n_micro)
                losses.append(loss)
                tokens += ntok
            for _ in range(n_micro - ga):
                # General questions with the original model's own answers, on top of the usual mix.
                rows = [replay_train[(rehearsed + j) % len(replay_train)] for j in range(mb)]
                rehearsed += mb
                tokens += b.train_microbatch(rows, 1.0 / n_micro)[1]
            cur_lr = lr_mult * lr_at(step, total, lr, cfg["warmup_frac"], cfg["min_lr_ratio"])
            gnorm = b.optimizer_step(cur_lr, cfg["max_grad_norm"])
            step += 1
            mean_loss = sum(losses) / max(1, len(losses))
            loss_ema = mean_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * mean_loss
            now = time.monotonic()
            tok_s = tokens / max(1e-6, now - t_last)
            metrics = {"loss": round(mean_loss, 4), "loss_ema": round(loss_ema, 4), "lr": cur_lr, "grad_norm": gnorm, "tok_s": round(tok_s, 1), "dev_pass": last_dev, "replay_dnll": last_dnll, "mem_gb": b.memory_gb()}
            if step % eval_every == 0 or step == total:
                dev_res = self._dev_eval(baseline, step)
                last_dev = dev_res["passed"] / max(dev_res["n"], 1)
                last_dnll = b.nll(holdout) - base_nll
                answers_dnll = b.nll(answers_only) - base_answers
                metrics.update(dev_pass=last_dev, replay_dnll=round(last_dnll, 4) if math.isfinite(last_dnll) else None)
                general_lost, general_note, kept_now = 0, "", 0
                if watch_general:
                    said = b.generate([row["messages"][:-1] for row in watch], max_new_tokens=16, temperature=0.0)
                    now_general = [general.check(row, o[0]) for row, o in zip(watch, said)]
                    gained, lost = discordant(base_general, now_general)
                    general_lost, kept_now = lost - gained, sum(now_general)
                    general_note = f", general questions {sum(now_general)}/{len(watch)} (the original: {sum(base_general)})"
                    metrics.update(general_pass=sum(now_general) / len(watch))
                events.log(f"{stage} step {step}/{total}: dev pass {last_dev:.0%}, drift {last_dnll:+.3f} nats/token (answers alone {answers_dnll:+.3f}){general_note}", "info")
                # Forgetting can only be judged when it could be measured before training started.
                why = gone_wrong(base_dev_pass, base_dev_format, dev_res, last_dnll if math.isfinite(base_nll) else 0.0, float(cfg["dnll_limit"]))
                over = watch_general and general_lost >= general_limit(len(watch))
                if why:
                    # Never keep, and never go on from, a model in this state: go back to the last
                    # good one (the best so far, or where the stage started) and tread more gently.
                    guard_trips += 1
                    st["why"] = why
                    good = best_dir if best_dir.is_dir() else (init_dir if init_dir.is_dir() else None)
                    events.log(f"Training went wrong at step {step}: {why}.", "warn")
                    if good is not None:
                        b.load_checkpoint(good)
                    if guard_trips > 2 or good is None:
                        stopped = "unstable"
                        events.log("Stopping this stage here" + ("." if good is None else ", with the best version so far." if best_dir.is_dir() else ", with the model as it was before the stage."), "warn")
                        break
                    lr_mult *= 0.5
                    if good == init_dir:
                        b.reset_optimizer()
                        step, patience, loss_ema, last_dev = 0, 0, None, None
                        st.update(step=0, best_dev=best, patience=0, lr_mult=lr_mult, guard_trips=guard_trips, last_dev=None)
                        self.state.save()
                        events.log("Starting this stage again, more gently (learning rate halved).", "warn")
                        # The sentence the window shows under the progress bar while this goes on.
                        events.stage("sft" if stage == "sft" else "rft", "started", "That attempt made the model worse, so training is starting again more gently")
                        prog.update(0, force=True)
                        continue
                    events.log("Went back to the best version so far; continuing more gently (learning rate halved).", "warn")
                    events.stage("sft" if stage == "sft" else "rft", "started", "Training slipped, so it went back to its best version and continues more gently")
                elif over:
                    # Learning the task is costing general skills beyond what the final exam accepts.
                    # Nothing is thrown away: from here on every step also rehearses general
                    # questions (more of them each time this happens) and moves more gently, and
                    # this version is not one of those the final choice is made from.
                    general_trips += 1
                    extra = min(ga, max(1, math.ceil(ga / 2)) * general_trips) if replay_train else 0
                    if general_trips <= 2:
                        lr_mult *= 0.5
                    events.log(f"It no longer answers {general_lost} of the {len(watch)} general questions it is checked on (more than {general_limit(len(watch)) - 1} is too close to what the final exam refuses): rehearsing them more from here on, and learning more gently.", "warn")
                    events.stage("sft" if stage == "sft" else "rft", "started", "Rehearsing general skills more while it learns your task")
                    if last_dev > spare + 1e-9 or (abs(last_dev - spare) <= 1e-9 and kept_now > spare_general):
                        spare, spare_general = last_dev, kept_now
                        b.save_checkpoint(spare_dir)
                else:
                    improved = last_dev > best + 1e-9
                    if improved or (watch_general and abs(last_dev - best) <= 1e-9 and kept_now > best_general):
                        best, best_general = last_dev, kept_now
                        b.save_checkpoint(best_dir)
                    if improved:
                        patience = 0
                    else:
                        patience += 1
                        if patience >= int(cfg["early_stop_patience"]):
                            stopped = "early_stop"
                            prog.update(total, metrics, force=True)
                            break
            if step % ckpt_every == 0 or step == total:
                b.save_checkpoint(ckpt)
                st.update(step=step, best_dev=best, spare_dev=spare, best_general=best_general, spare_general=spare_general, patience=patience, lr_mult=lr_mult, guard_trips=guard_trips, general_trips=general_trips, extra_rehearsal=extra, last_dev=last_dev)
                self.state.save()
            prog.update(step, metrics)
            if self.over_budget(0.75 if stage == "sft" else 0.9):
                events.log("Time budget reached; finishing with the best result so far", "warn")
                stopped = "budget"
                break
        kept_general = True
        if best_dir.is_dir():
            b.load_checkpoint(best_dir)
        elif spare_dir.is_dir():
            # No version kept its general skills within the limit. The one that learned the task
            # best goes to the final exam, which measures both and says what it finds.
            b.load_checkpoint(spare_dir)
            best, best_general, kept_general = spare, spare_general, False
            events.log("No version of this stage kept its general skills within the limit; the one that learned your task best goes to the final exam.", "warn")
        st.update(step=step, best_dev=best, stopped=stopped, guard_trips=guard_trips, general_trips=general_trips)
        self.state.save()
        return {"best_dev": best, "steps": step, "total": total, "stopped": stopped, "examples": len(examples), "replay": n_replay, "went_wrong": guard_trips, "lr_mult": lr_mult, "why": st.get("why"), "rehearsed_more": general_trips, "kept_general": kept_general, "general_answered": best_general if watch_general else None, "general_watched": len(watch) if watch_general else None}

    # ---------------------------------------------------------------- stage 2 --------------
    def rft(self, replay_rows: list[dict], baseline: dict, best_dev: float, placebo: bool = False) -> dict:
        b = self.backend
        s2 = self.plan["stage2"]
        prompts = self.train[: min(len(self.train), 256)]
        o0, o1 = OVERALL["stage2"]
        history = []
        current = best_dev
        keep_dir = self.run_dir / "best-sft"
        for rnd in range(int(s2["rounds"])):
            if current >= 0.999 or self.over_budget(0.85):
                break
            events.stage("rft", "started", f"Round {rnd + 1}: practicing and keeping only answers that pass your checks")
            k = int(s2["k"])
            samples = b.generate([it["messages"] for it in prompts], max_new_tokens=self.max_new, temperature=float(s2["temperature"]), top_p=0.95, n=k, seed=self.plan["seed"] + rnd)
            flat_items = [it for it in prompts for _ in range(k)]
            flat_outs = [o for outs in samples for o in outs]
            scored = score_batch(flat_items, flat_outs, self.vset, self.ctx)
            accepted: list[dict] = []
            per_prompt: dict[str, int] = {}
            seen: set[tuple[str, str]] = set()
            pass_flags = [r["pass"] for r in scored]
            accept_rate = sum(pass_flags) / max(1, len(pass_flags))
            rng = random.Random(self.plan["seed"] + 99 + rnd)
            for it, out, ok in zip(flat_items, flat_outs, pass_flags):
                chosen = (rng.random() < accept_rate) if placebo else ok
                key = (it["item_id"], out.strip())
                if not chosen or key in seen or per_prompt.get(it["item_id"], 0) >= 2:
                    continue
                seen.add(key)
                per_prompt[it["item_id"]] = per_prompt.get(it["item_id"], 0) + 1
                accepted.append({"messages": it["messages"] + [{"role": "assistant", "content": out}]})
            solved = {it["item_id"] for it, ok in zip(flat_items, pass_flags) if ok}
            for it in prompts:  # never-solved prompts keep their reference answer
                if it["item_id"] not in solved:
                    accepted.append(to_example(it))
            events.log(f"Round {rnd + 1}: {sum(pass_flags)}/{len(pass_flags)} practice answers passed ({accept_rate:.0%}); training on {len(accepted)}", "info")
            if not accepted:
                break
            b.save_checkpoint(self.run_dir / "pre-rft")
            b.reset_optimizer()
            res = self.sft(accepted, replay_rows, baseline, stage=f"rft{rnd + 1}{'-placebo' if placebo else ''}", lr=float(s2["lr"]), epochs=1, overall=(o0 + (o1 - o0) * rnd / max(1, s2["rounds"]), o0 + (o1 - o0) * (rnd + 1) / max(1, s2["rounds"])), resume=False)
            gain = 100 * (res["best_dev"] - current)
            history.append({"round": rnd + 1, "accept_rate": accept_rate, "dev": res["best_dev"], "gain_pts": gain})
            if not res.get("kept_general", True):
                # Every version of this round lost general skills beyond the limit: not worth it.
                b.load_checkpoint(self.run_dir / "pre-rft")
                events.stage("rft", "completed", f"Round {rnd + 1} cost general skills; kept the previous version")
                break
            if gain >= float(s2["min_gain_pts"]):
                current = res["best_dev"]
                keep_dir = self.run_dir / f"best-rft{rnd + 1}{'-placebo' if placebo else ''}"
                events.stage("rft", "completed", f"Round {rnd + 1} improved practice score by {gain:.1f} points")
            else:
                b.load_checkpoint(self.run_dir / "pre-rft")
                events.stage("rft", "completed", f"Round {rnd + 1} did not help ({gain:+.1f} points); kept the previous version")
                break
        return {"rounds": history, "best_dev": current, "kept": keep_dir.name}

    def grpo(self, baseline: dict, best_dev: float) -> Optional[dict]:
        b = self.backend
        if not hasattr(b, "grpo"):
            return None
        try:
            import trl  # noqa: F401,WPS433
        except ImportError:
            events.log("TRL is not installed; using outcome-filtered refinement instead of GRPO", "warn")
            return None
        s2 = self.plan["stage2"]
        prompts = self.train[: min(len(self.train), 512)]

        def reward(indices: list[int], completions: list[str]) -> list[float]:
            items = [prompts[i] for i in indices]
            return [r["score"] if r["pass"] or r["score"] else 0.0 for r in score_batch(items, completions, self.vset, self.ctx)]

        steps = max(10, min(300, len(prompts) // 2))
        o0, o1 = OVERALL["stage2"]
        prog = events.Progress("grpo", steps, o0, o1)
        b.save_checkpoint(self.run_dir / "pre-grpo")
        info = b.grpo([it["messages"] for it in prompts], reward, {"lr": 1e-5, "num_generations": max(2, int(s2["k"])), "max_completion_length": self.max_new, "max_steps": steps, "temperature": 1.0, "beta": 0.02}, self.run_dir / "grpo", prog)
        dev = self._dev_eval(baseline, steps)
        if 100 * (dev - best_dev) >= float(s2["min_gain_pts"]):
            b.save_checkpoint(self.run_dir / "best-grpo")
            return {"best_dev": dev, "kept": "best-grpo", **info}
        b.load_checkpoint(self.run_dir / "pre-grpo")
        return {"best_dev": best_dev, "kept": "best-sft", **info}

    # ---------------------------------------------------------------- gate -----------------
    def gate(self, baseline: dict, candidate_dir: Path, placebo_result: Optional[dict] = None) -> dict:
        o0, o1 = OVERALL["gate"]
        prog = events.Progress("gate", len(self.test) * 2, o0, o1)
        b = self.load(candidate_dir)
        cand = evaluate(b, self.test, self.vset, self.ctx, self.max_new, progress=prog)
        emit_eval("test", "candidate", cand)
        cand_gen = eval_general(b, int(self.gate_cfg["general_suite_n"]))
        emit_eval("general", "candidate", cand_gen)

        incumbent = active_version(self.root)
        ref_name, ref = "base", baseline["test"]
        ref_gen = baseline["general"]
        if incumbent:
            inc_dir = self.paths.versions / incumbent["version"] / "adapter"
            if inc_dir.is_dir():
                cache = self._cache_key("version:" + incumbent["adapter_sha256"])
                if cache.is_file():
                    inc = json.loads(cache.read_text())
                else:
                    bi = self.load(inc_dir)
                    r = evaluate(bi, self.test, self.vset, self.ctx, self.max_new, progress=prog, offset=len(self.test))
                    inc = {"n": r["n"], "passed": r["passed"], "pass": r["pass"], "format_rate": format_rate(r), "format_pass": format_pass(r), "general": eval_general(bi, int(self.gate_cfg["general_suite_n"]))}
                    atomic_write_json(cache, inc)
                emit_eval("test", "incumbent", inc)
                ref_name, ref, ref_gen = f"current version {incumbent['version']}", inc, inc["general"]
                self.load(candidate_dir)
        prog.update(prog.total, force=True)

        n = len(self.test)
        b_cnt, c_cnt = discordant(ref["pass"], cand["pass"])
        p_val = mcnemar_exact(b_cnt, c_cnt)
        diff, lo, hi = paired_bootstrap(ref["pass"], cand["pass"])
        base_rate = ref["passed"] / max(1, n)
        cand_rate = cand["passed"] / max(1, n)
        delta = 100 * (cand_rate - base_rate)
        g_n = cand_gen["n"]
        g_b, g_c = discordant(ref_gen["pass"], cand_gen["pass"])
        # What exactly was lost: the general questions the reference answered and this model no
        # longer does, with what it says now (these are stock questions, nothing of the user's).
        lost_general = [i for i, (was, now) in enumerate(zip(ref_gen["pass"], cand_gen["pass"])) if was and not now]
        if lost_general:
            stock = general.suite(g_n)
            for i in lost_general[:8]:
                said = " ".join(str((cand_gen.get("outputs") or [""] * g_n)[i]).split())
                events.log(f"General question no longer answered correctly: {stock[i]['messages'][-1]['content'][:90]} -> {said[:90]}", "info")
        g_drop = 100 * (ref_gen["passed"] - cand_gen["passed"]) / max(1, g_n)
        g_p = mcnemar_exact(g_b, g_c)
        f_ref, f_cand = ref.get("format_rate"), format_rate(cand)
        aud = audit_rate(cand)
        decision, label = "REJECT", None
        if n < int(self.gate_cfg["min_test_items"]) // 2 or n < 10:
            decision = "INSUFFICIENT_DATA"
        verified = n >= int(self.gate_cfg["verified_min_items"])
        task_ok = delta >= float(self.gate_cfg["min_delta_pts"]) and ((lo > 0) if verified else (p_val < 0.05))
        general_ok = not (g_drop > float(self.gate_cfg["max_general_drop_pts"]) and g_p <= 0.10)
        format_ok = format_check(f_ref, f_cand, ref.get("format_pass"), format_pass(cand), task_gain=cand["passed"] - ref["passed"])
        audit_ok = True if aud is None else aud >= 0.9
        # Training broke at every attempt and nothing better than the starting point exists: what
        # was just tested is the untrained model, which must never be saved as a "version".
        nothing_trained = training_failed(self.training_info)
        checks = {"task": task_ok, "general": general_ok, "format": format_ok, "audit": audit_ok, "leakage": True, "training": not nothing_trained}
        if decision != "INSUFFICIENT_DATA":
            decision = "PROMOTE" if all(checks.values()) else "REJECT"
            label = "VERIFIED" if verified else "PRELIMINARY"

        lines = [f"Your trained model passed {cand_rate:.0%} of the {n} hidden test questions, versus {base_rate:.0%} for the {ref_name} ({delta:+.1f} points)."]
        if decision == "PROMOTE":
            lines.append("That difference is statistically solid" + (f" (p = {p_val:.2g})." if not verified else f" (95% range {lo:+.1f} to {hi:+.1f} points)."))
            lines.append("General skills: no meaningful drop." if g_drop <= 1 else f"General skills dipped {g_drop:.1f} points, within the allowed limit.")
            if f_ref is not None and f_cand is not None and f_cand < f_ref - 0.02:
                lines.append(f"About {round(100 * (1 - f_cand))}% of its answers were not well-formed (the {ref_name}: {round(100 * (1 - f_ref))}%); those count as wrong in the score above.")
            if label == "PRELIMINARY":
                lines.append(f"Marked preliminary: add more examples to reach {self.gate_cfg['verified_min_items']}+ hidden questions for a verified result.")
        elif decision == "INSUFFICIENT_DATA":
            lines.append("There are too few hidden test questions to judge fairly. Add more examples and train again.")
        elif nothing_trained:
            kept = "Your current version stays active." if ref_name != "base" else "Nothing was saved, and the original model is unchanged."
            why_last = (self.training_info or {}).get("why")
            lines = [f"Training did not hold steady on this computer: it was tried three times, each time more gently, and each time the model got worse instead of better{f' (the last time: {why_last})' if why_last else ''}. So there is no trained model to judge (the model as it was before training passes {cand_rate:.0%} of the {n} hidden test questions). {kept}"]
        else:
            why = []
            if not task_ok:
                why.append(f"it did worse than the {ref_name} on the hidden questions" if (delta < 0 and p_val < 0.05) else "the improvement is too small or not statistically reliable yet")
            if not general_ok:
                why.append(f"general skills dropped {g_drop:.1f} points")
            if not format_ok:
                why.append("it formats answers correctly less often than before")
            if not audit_ok:
                why.append("some correct-looking answers fail on scrambled data (likely memorized)")
            kept = "Your current version stays active." if ref_name != "base" else "Nothing was saved, and the original model is unchanged."
            lines.append("Not promoted because " + "; ".join(why) + ". " + kept)

        report = {
            "version": 1,
            "run_id": self.plan["run_id"],
            "created_at": time.time(),
            "decision": decision,
            "label": label,
            "compared_against": ref_name,
            "test": {"n": n, "merkle_root": json.loads(self.paths.test_manifest.read_text())["merkle_root"], "baseline_pass_rate": base_rate, "candidate_pass_rate": cand_rate, "baseline_ci95": wilson(ref["passed"], n), "candidate_ci95": wilson(cand["passed"], n), "delta_pts": delta, "mcnemar": {"b": b_cnt, "c": c_cnt, "p": p_val}, "bootstrap_pts": {"mean": diff, "lo95": lo, "hi95": hi}},
            "base_model": {"test_pass_rate": baseline["test"]["passed"] / max(1, n)},
            "general": {"n": g_n, "baseline": ref_gen["passed"], "candidate": cand_gen["passed"], "drop_pts": g_drop, "mcnemar_p": g_p},
            "format": {"baseline": f_ref, "candidate": f_cand},
            "audit_pass_rate": aud,
            "placebo": placebo_result,
            "training": None if not self.training_info else {k: self.training_info.get(k) for k in ("steps", "total", "stopped", "best_dev", "went_wrong", "lr_mult", "why", "rehearsed_more", "kept_general", "general_answered", "general_watched")},
            "checks": checks,
            "explanation": " ".join(lines),
            "plan": {k: self.plan[k] for k in ("backend", "base_model", "quant", "lora", "sft", "stage2", "tier") if k in self.plan},
            "elapsed_minutes": round(self.elapsed_min(), 2),
            "note": "Hidden test questions are never copied into reports or shown in the app.",
        }
        report_path = self.run_dir / "report.json"
        atomic_write_json(report_path, report)
        return {"report": report, "report_path": report_path, "decision": decision, "label": label, "delta": delta, "base_rate": base_rate, "cand_rate": cand_rate, "checks": checks}

    def promote(self, candidate_dir: Path, gate: dict) -> dict:
        self.paths.versions.mkdir(parents=True, exist_ok=True)
        existing = sorted(int(p.name[1:]) for p in self.paths.versions.glob("v*") if p.name[1:].isdigit())
        version = f"v{(existing[-1] + 1) if existing else 1}"
        vdir = self.paths.versions / version
        adir = vdir / "adapter"
        shutil.copytree(candidate_dir, adir)
        for junk in ("optimizer.pt", "optimizer.safetensors"):
            (adir / junk).unlink(missing_ok=True)
        if self.backend is not None:
            self.backend.write_adapter_meta(adir)
        digest = sha256_dir(adir)
        report = gate["report"] | {"promoted_version": version, "adapter_sha256": digest}
        atomic_write_json(vdir / "report.json", report)
        prev = active_version(self.root)
        atomic_write_json(self.paths.active, {"version": version, "adapter_sha256": digest, "run_id": self.plan["run_id"], "previous": prev["version"] if prev else None, "promoted_at": time.time()})
        project = load_project(self.root)
        project["active_version"] = version
        save_project(self.root, project)
        events.emit("artifact", kind="adapter", path=str(adir), sha256=digest)
        exports = self.export(adir, vdir)
        return {"version": version, "adapter_dir": str(adir), "adapter_sha256": digest, "exports": exports}

    def export(self, adir: Path, vdir: Path) -> dict:
        from engine.adapters import try_export_gguf  # noqa: WPS433

        out: dict[str, Any] = {"adapter": str(adir)}
        params_b = (self.plan.get("memory_estimate") or {}).get("params_b") or 0
        if self.plan["backend"] in ("torch", "unsloth") and not self.plan["base_model"].get("gguf") and (self.plan.get("export_merged") or 0 < params_b <= 1.5 or self.plan.get("tier") == "smoke"):
            try:
                from engine.backends.torch_backend import write_merged  # noqa: WPS433

                mdir = write_merged(self.plan, adir, vdir / "merged")
                out["merged"] = str(mdir)
                events.emit("artifact", kind="merged", path=str(mdir), sha256=sha256_dir(mdir))
            except Exception as exc:  # noqa: BLE001
                events.log(f"Merged export skipped: {exc}", "warn")
        # (Not for MLX: the converter needs PyTorch and a model in the Hugging Face layout, neither of
        # which a Mac build has. Trying anyway put a Python traceback into the log of every run.)
        gguf = None if self.plan["backend"] == "mlx" else try_export_gguf(self.plan, adir, vdir / "gguf")
        if gguf:
            out["gguf"] = str(gguf)
            events.emit("artifact", kind="gguf", path=str(gguf), sha256=sha256_dir(gguf.parent))
        return out


# ------------------------------------------------------------------------------------------
# Entry points
# ------------------------------------------------------------------------------------------


def _stage(state: RunState, name: str, detail: str) -> bool:
    if state.done(name):
        events.stage(name, "skipped", "already finished (resumed run)")
        return False
    events.stage(name, "started", detail)
    return True


HEAVY_RUN_DIRS = ("ckpt-", "best-", "init-", "pre-", "candidate")


def prune_run_dirs(runs: Path, keep: Optional[str] = None) -> int:
    """Delete checkpoints of earlier runs; their small records (plan, state, report) stay.

    Every run writes several copies of the adapter (checkpoints, the best so far, the candidate).
    Without this, each training run, finished or abandoned, left hundreds of megabytes behind.
    """
    freed = 0
    if not runs.is_dir():
        return 0
    for run in runs.iterdir():
        if not run.is_dir() or run.name == keep:
            continue
        for child in run.iterdir():
            if child.is_dir() and child.name.startswith(HEAVY_RUN_DIRS):
                freed += sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
                shutil.rmtree(child, ignore_errors=True)
    return freed


def run_train(root: Path, plan: dict, resume: bool = False) -> dict:
    tr = Trainer(root, plan)
    st = tr.state
    freed = prune_run_dirs(tr.paths.runs, keep=plan["run_id"])
    if freed > 1e6:
        events.log(f"Removed {freed / 1e6:.0f} MB of checkpoints left by earlier runs", "info")
    events.stage("verify_data", "started", "Checking that no hidden test question leaked into training")
    tr.verify_data()
    events.stage("verify_data", "completed")

    tr.load()
    if _stage(st, "baseline", "Measuring the original model on the hidden test"):
        baseline = tr.baseline()
        st.complete("baseline", pass_rate=baseline["test"]["passed"] / max(1, baseline["test"]["n"]))
        events.stage("baseline", "completed", f"Original model: {baseline['test']['passed']}/{baseline['test']['n']} correct")
    else:
        baseline = tr.baseline()

    replay_rows = tr.replay()
    if not st.done("replay"):
        st.complete("replay", n=len(replay_rows))

    best_dir = tr.run_dir / "best-sft"
    sft_info = st.data.get("info", {}).get("sft")
    if _stage(st, "sft", f"Learning from {len(tr.train)} examples"):
        tr.backend.prepare_training()
        sft_info = tr.sft([to_example(it) for it in tr.train], replay_rows, baseline, stage="sft", lr=float(plan["sft"]["lr"]), epochs=float(plan["sft"]["epochs"]), overall=OVERALL["sft"], resume=resume)
        if not best_dir.is_dir():
            tr.backend.save_checkpoint(best_dir)
        st.complete("sft", **sft_info)
        events.stage("sft", "completed", f"Best practice score {max(0.0, sft_info['best_dev']):.0%} after {sft_info['steps']} steps ({sft_info['stopped']})")
    tr.training_info = sft_info

    chosen = best_dir
    s2 = plan["stage2"]
    placebo_result = None
    # A first stage that never produced a usable model leaves nothing to refine, and one whose
    # best version already lost too many general questions would only lose more by practicing.
    if s2["mode"] != "none" and not training_failed(sft_info) and (sft_info or {}).get("kept_general", True) and _stage(st, "stage2", "Practicing with automatic grading"):
        if not tr.backend.has_adapter:
            tr.load(best_dir, trainable=True)
            tr.backend.prepare_training()
        else:
            tr.backend.load_checkpoint(best_dir)
        best_dev = float((sft_info or {}).get("best_dev", 0.0))
        info = None
        if s2["mode"] == "grpo":
            info = tr.grpo(baseline, best_dev)
        if info is None:
            info = tr.rft(replay_rows, baseline, best_dev)
        chosen = tr.run_dir / info["kept"]
        if s2.get("placebo"):
            tr.load(best_dir, trainable=True)
            tr.backend.prepare_training()
            placebo = tr.rft(replay_rows, baseline, best_dev, placebo=True)
            placebo_result = {"best_dev": placebo["best_dev"], "kept": placebo["kept"]}
        st.complete("stage2", **{k: v for k, v in info.items() if k != "rounds"}, rounds=info.get("rounds"))
        events.stage("stage2", "completed", f"Practice score now {info['best_dev']:.0%}")
    elif st.done("stage2"):
        chosen = tr.run_dir / st.data["info"]["stage2"]["kept"]
    if not chosen.is_dir():
        chosen = best_dir

    # The gate always evaluates a clean export of the chosen adapter (what would ship).
    tr.load(chosen)
    cand_dir = tr.run_dir / "candidate"
    if cand_dir.exists():
        shutil.rmtree(cand_dir)
    tr.backend.save_adapter(cand_dir)

    events.stage("gate", "started", "Final exam: original vs trained on hidden questions")
    gate = tr.gate(baseline, cand_dir, placebo_result)
    events.emit(
        "gate",
        decision=gate["decision"],
        label=gate["label"],
        delta_pts=round(gate["delta"], 2),
        baseline_pass_rate=round(gate["base_rate"], 4),
        candidate_pass_rate=round(gate["cand_rate"], 4),
        checks=gate["checks"],
        version=None,
        report_path=str(gate["report_path"]),
        explanation=gate["report"]["explanation"],
    )
    events.stage("gate", "completed", gate["decision"])
    result: dict[str, Any] = {"decision": gate["decision"], "label": gate["label"], "report_path": str(gate["report_path"]), "run_id": plan["run_id"], "delta_pts": gate["delta"], "explanation": gate["report"]["explanation"]}
    if gate["decision"] == "PROMOTE":
        events.stage("promote", "started", "Saving your new version")
        promoted = tr.promote(cand_dir, gate)
        result.update(promoted)
        events.emit("gate", decision="PROMOTE", label=gate["label"], delta_pts=round(gate["delta"], 2), baseline_pass_rate=round(gate["base_rate"], 4), candidate_pass_rate=round(gate["cand_rate"], 4), checks=gate["checks"], version=promoted["version"], report_path=str(tr.paths.versions / promoted["version"] / "report.json"), explanation=gate["report"]["explanation"])
        events.stage("promote", "completed", promoted["version"])
    else:
        active = active_version(root)
        result["active_version"] = active["version"] if active else None
        events.log("Kept the current version active (automatic rollback)." if active else "Nothing was saved; the original model is unchanged.", "info")
    events.Progress("done", 1, 100, 100).update(1, force=True)
    st.complete("gate", decision=gate["decision"])
    # The run is over: its checkpoints are of no further use (a promoted adapter lives in versions/).
    if tr.backend is not None:
        tr.backend.unload()
    for child in tr.run_dir.iterdir():
        if child.is_dir() and child.name.startswith(HEAVY_RUN_DIRS):
            shutil.rmtree(child, ignore_errors=True)
    return result


def run_gate_only(root: Path, plan: dict, adapter_dir: Path) -> dict:
    """Gate an externally produced adapter (import / cloud) against the incumbent."""
    tr = Trainer(root, plan)
    tr.verify_data()
    tr.load()
    baseline = tr.baseline()
    gate = tr.gate(baseline, adapter_dir)
    events.emit("gate", decision=gate["decision"], label=gate["label"], delta_pts=round(gate["delta"], 2), baseline_pass_rate=round(gate["base_rate"], 4), candidate_pass_rate=round(gate["cand_rate"], 4), checks=gate["checks"], version=None, report_path=str(gate["report_path"]), explanation=gate["report"]["explanation"])
    result: dict[str, Any] = {"decision": gate["decision"], "explanation": gate["report"]["explanation"], "report_path": str(gate["report_path"])}
    if gate["decision"] == "PROMOTE":
        result.update(tr.promote(adapter_dir, gate))
    return result
