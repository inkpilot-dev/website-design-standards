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
    return {"pass": passes, "n": len(items), "passed": sum(passes)}


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
        if len(self.test) < int(self.gate_cfg["min_test_items"]):
            events.log(f"Only {len(self.test)} hidden test questions; the result will be marked preliminary or insufficient.", "warn")
        if not self.train:
            raise ValueError("No training examples with answers were found.")
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
            "test": {k: test[k] for k in ("n", "passed", "pass")} | {"format_rate": format_rate(test), "outputs": test["outputs"]},
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
        prompts = general.replay_prompts(64)
        outs = b.generate([p["messages"] for p in prompts], max_new_tokens=64, temperature=0.0, use_adapter=False)
        rows = [{"messages": p["messages"] + [{"role": "assistant", "content": o[0] or "OK"}]} for p, o in zip(prompts, outs)]
        write_jsonl(path, rows)
        return rows

    # ---------------------------------------------------------------- SFT loop -------------
    def _dev_eval(self, baseline: dict, step: int) -> float:
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
        return res["passed"] / max(res["n"], 1)

    def sft(self, examples: list[dict], replay_rows: list[dict], baseline: dict, *, stage: str, lr: float, epochs: float, overall: tuple[float, float], resume: bool) -> dict:
        b = self.backend
        cfg = self.plan["sft"]
        holdout = replay_rows[: max(4, len(replay_rows) // 4)]
        replay_train = replay_rows[len(holdout) :]
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
        step = 0
        best = st.get("best_dev", -1.0)
        patience = st.get("patience", 0)
        lr_mult = st.get("lr_mult", 1.0)
        guard_trips = st.get("guard_trips", 0)
        if resume and ckpt.is_dir() and st.get("step"):
            b.load_checkpoint(ckpt)
            step = int(st["step"])
            events.log(f"Resumed {stage} from step {step}", "info")
        base_nll = st.get("base_nll")
        if base_nll is None:
            base_nll = b.nll(holdout)
            st["base_nll"] = base_nll
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
            for k in range(ga):
                idx = order[pos + k * mb : pos + (k + 1) * mb]
                if not idx:
                    idx = order[: mb]
                loss, ntok = b.train_microbatch([data[i] for i in idx], 1.0 / ga)
                losses.append(loss)
                tokens += ntok
            cur_lr = lr_mult * lr_at(step, total, lr, cfg["warmup_frac"], cfg["min_lr_ratio"])
            gnorm = b.optimizer_step(cur_lr, cfg["max_grad_norm"])
            step += 1
            mean_loss = sum(losses) / max(1, len(losses))
            loss_ema = mean_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * mean_loss
            now = time.monotonic()
            tok_s = tokens / max(1e-6, now - t_last)
            metrics = {"loss": round(mean_loss, 4), "loss_ema": round(loss_ema, 4), "lr": cur_lr, "grad_norm": gnorm, "tok_s": round(tok_s, 1), "dev_pass": last_dev, "replay_dnll": last_dnll, "mem_gb": b.memory_gb()}
            if step % eval_every == 0 or step == total:
                last_dev = self._dev_eval(baseline, step)
                last_dnll = b.nll(holdout) - base_nll
                metrics.update(dev_pass=last_dev, replay_dnll=round(last_dnll, 4))
                events.log(f"{stage} step {step}/{total}: dev pass {last_dev:.0%}, drift {last_dnll:+.3f} nats/token", "info")
                if last_dnll > float(cfg["dnll_limit"]):
                    guard_trips += 1
                    events.log(f"Forgetting guard tripped (drift {last_dnll:.3f} > {cfg['dnll_limit']}).", "warn")
                    if guard_trips >= 2 or not best_dir.is_dir():
                        stopped = "forgetting_guard"
                        break
                    b.load_checkpoint(best_dir)
                    lr_mult *= 0.5
                elif last_dev > best + 1e-9:
                    best, patience = last_dev, 0
                    b.save_checkpoint(best_dir)
                else:
                    patience += 1
                    if patience >= int(cfg["early_stop_patience"]):
                        stopped = "early_stop"
                        prog.update(total, metrics, force=True)
                        break
            if step % ckpt_every == 0 or step == total:
                b.save_checkpoint(ckpt)
                st.update(step=step, best_dev=best, patience=patience, lr_mult=lr_mult, guard_trips=guard_trips, last_dev=last_dev)
                self.state.save()
            prog.update(step, metrics)
            if self.over_budget(0.75 if stage == "sft" else 0.9):
                events.log("Time budget reached; finishing with the best result so far", "warn")
                stopped = "budget"
                break
        if best_dir.is_dir():
            b.load_checkpoint(best_dir)
        st.update(step=step, best_dev=best, stopped=stopped)
        self.state.save()
        return {"best_dev": best, "steps": step, "total": total, "stopped": stopped, "examples": len(examples), "replay": n_replay}

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
                    inc = {"n": r["n"], "passed": r["passed"], "pass": r["pass"], "format_rate": format_rate(r), "general": eval_general(bi, int(self.gate_cfg["general_suite_n"]))}
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
        format_ok = True if (f_ref is None or f_cand is None) else (f_cand >= f_ref - 0.02)
        audit_ok = True if aud is None else aud >= 0.9
        checks = {"task": task_ok, "general": general_ok, "format": format_ok, "audit": audit_ok, "leakage": True}
        if decision != "INSUFFICIENT_DATA":
            decision = "PROMOTE" if all(checks.values()) else "REJECT"
            label = "VERIFIED" if verified else "PRELIMINARY"

        lines = [f"Your trained model passed {cand_rate:.0%} of the {n} hidden test questions, versus {base_rate:.0%} for the {ref_name} ({delta:+.1f} points)."]
        if decision == "PROMOTE":
            lines.append("That difference is statistically solid" + (f" (p = {p_val:.2g})." if not verified else f" (95% range {lo:+.1f} to {hi:+.1f} points)."))
            lines.append("General skills: no meaningful drop." if g_drop <= 1 else f"General skills dipped {g_drop:.1f} points, within the allowed limit.")
            if label == "PRELIMINARY":
                lines.append(f"Marked preliminary: add more examples to reach {self.gate_cfg['verified_min_items']}+ hidden questions for a verified result.")
        elif decision == "INSUFFICIENT_DATA":
            lines.append("There are too few hidden test questions to judge fairly. Add more examples and train again.")
        else:
            why = []
            if not task_ok:
                why.append("the improvement is too small or not statistically reliable yet")
            if not general_ok:
                why.append(f"general skills dropped {g_drop:.1f} points")
            if not format_ok:
                why.append("it formats answers correctly less often than before")
            if not audit_ok:
                why.append("some correct-looking answers fail on scrambled data (likely memorized)")
            lines.append("Not promoted because " + "; ".join(why) + ". Your current version stays active.")

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
        if self.plan["backend"] in ("torch", "unsloth") and (self.plan.get("export_merged") or 0 < params_b <= 1.5 or self.plan.get("tier") == "smoke"):
            try:
                from engine.backends.torch_backend import write_merged  # noqa: WPS433

                mdir = write_merged(self.plan, adir, vdir / "merged")
                out["merged"] = str(mdir)
                events.emit("artifact", kind="merged", path=str(mdir), sha256=sha256_dir(mdir))
            except Exception as exc:  # noqa: BLE001
                events.log(f"Merged export skipped: {exc}", "warn")
        gguf = try_export_gguf(self.plan, adir, vdir / "gguf")
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


def run_train(root: Path, plan: dict, resume: bool = False) -> dict:
    tr = Trainer(root, plan)
    st = tr.state
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

    chosen = best_dir
    s2 = plan["stage2"]
    placebo_result = None
    if s2["mode"] != "none" and _stage(st, "stage2", "Practicing with automatic grading"):
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
        events.log("Kept the current version active (automatic rollback).", "info")
    events.Progress("done", 1, 100, 100).update(1, force=True)
    st.complete("gate", decision=gate["decision"])
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
