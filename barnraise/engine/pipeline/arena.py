"""Proof Arena: the original model and your trained model answer the same questions side by side.

Questions come from the dev split (never the hidden test set) or from the user's own prompt.
Both answers are graded by the same verifiers, and each case is emitted as a `benchmark` event.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from engine import events
from engine.backends.base import make_backend
from engine.contract import ProjectPaths, load_project, read_jsonl
from engine.inference_worker import default_system_prompt, serving_plan
from engine.textnorm import prompt_hash
from engine.verify.core import Context, load_verifier_set, score_item


def run_arena(root: Path, prompt: Optional[str] = None, n: int = 4, max_new_tokens: int = 256) -> dict:
    load_project(root)
    plan, adapter, version = serving_plan(root)
    if adapter is None or not adapter.is_dir():
        raise ValueError("Train a model first: there is no trained version to compare yet.")
    vset = load_verifier_set(root)
    ctx = Context(root=root)
    if prompt:
        system = default_system_prompt(root)
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        cases = [{"item_id": prompt_hash(prompt), "messages": msgs, "reference": None, "reference_json": None, "verifiers": [v["id"] for v in vset["verifiers"]], "meta": {}}]
    else:
        pool = read_jsonl(ProjectPaths(root).split("dev")) or read_jsonl(ProjectPaths(root).split("train"))
        cases = pool[:n]
    events.stage("arena", "started", f"Comparing on {len(cases)} question(s)")
    backend = make_backend(plan, inference_only=True)
    backend.load(adapter)
    prog = events.Progress("arena", len(cases), 0, 100)
    wins = {"trained": 0, "baseline": 0, "tie": 0}
    for i, case in enumerate(cases):
        base_out = backend.generate([case["messages"]], max_new_tokens, use_adapter=False)[0][0]
        ctx.baseline_output = base_out
        trained_out = backend.generate([case["messages"]], max_new_tokens, use_adapter=True)[0][0]
        b = score_item(case, base_out, vset, ctx)
        t = score_item(case, trained_out, vset, ctx)
        has_checks = any(v["pass"] is not None for v in t["verdicts"])
        if has_checks:
            wins["trained" if t["score"] > b["score"] else "baseline" if b["score"] > t["score"] else "tie"] += 1
        user = next(m["content"] for m in reversed(case["messages"]) if m["role"] == "user")
        events.emit(
            "benchmark",
            source="arena",
            case_id=case["item_id"],
            prompt=user,
            reference=case.get("reference"),
            baseline={"output": base_out, "pass": b["pass"] if has_checks else None, "score": b["score"] if has_checks else None},
            trained={"output": trained_out, "pass": t["pass"] if has_checks else None, "score": t["score"] if has_checks else None},
            step=None,
            judge=None,
        )
        prog.update(i + 1, force=True)
    events.stage("arena", "completed")
    return {"cases": len(cases), "wins": wins, "version": version}
