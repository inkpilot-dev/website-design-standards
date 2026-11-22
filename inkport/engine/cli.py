"""`python -m engine <command>`: the sidecar entry point the Rust supervisor invokes.

Every command writes NDJSON events (contracts/schemas/event.schema.json) to stdout and ends
with exactly one `done` event. Exit code 0 = ok, 1 = handled error (an `error` event
explains it), 2 = bad arguments.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from engine import __version__, events


def _backends() -> dict[str, bool]:
    import importlib.util as iu  # noqa: WPS433

    found = {name: iu.find_spec(mod) is not None for name, mod in {"torch": "torch", "mlx": "mlx", "unsloth": "unsloth", "trl": "trl", "llama_cpp": "llama_cpp", "peft": "peft", "transformers": "transformers"}.items()}
    found["cuda"] = found["mps"] = False
    if found["torch"]:
        try:
            import torch  # noqa: WPS433

            found["cuda"] = bool(torch.cuda.is_available())
            found["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        except Exception:  # noqa: BLE001
            pass
    return found


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:48] or "project"


# ------------------------------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------------------------------


def cmd_hello(a: argparse.Namespace) -> dict:
    events.emit("hello", engine_version=__version__, python=platform.python_version(), platform=f"{sys.platform}-{platform.machine()}", backends=_backends())
    return {"engine_version": __version__}


def cmd_detect(a: argparse.Namespace) -> dict:
    from engine.hardware_profile import detect  # noqa: WPS433

    report = detect()
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
    return {"hardware": report}


def _load_hardware(path: str | None) -> dict:
    from engine.hardware_profile import detect  # noqa: WPS433

    if path:
        report = json.loads(Path(path).read_text())
        if "recommendation" not in report:
            from engine.hardware_profile import select_tier  # noqa: WPS433

            report["recommendation"] = select_tier(report)
        return report
    return detect()


def _make_plan(a: argparse.Namespace) -> dict:
    from engine.contract import load_project  # noqa: WPS433
    from engine.hardware_profile import make_plan  # noqa: WPS433

    project = load_project(Path(a.project))
    overrides = json.loads(a.overrides) if getattr(a, "overrides", None) else None
    return make_plan(project, _load_hardware(getattr(a, "hardware", None)), preset=a.preset, overrides=overrides)


def cmd_create_project(a: argparse.Namespace) -> dict:
    from engine.contract import load_contract, save_project, template_meta  # noqa: WPS433

    template_meta(a.template)
    root = Path(a.root) / slugify(a.name)
    if (root / "project.json").exists():
        raise ValueError(f"a project named '{a.name}' already exists")
    models = load_contract("tiers.json")["models"]
    project = {
        "id": uuid.uuid4().hex[:12],
        "slug": slugify(a.name),
        "name": a.name,
        "template": a.template,
        "created_at": time.time(),
        "base_model": {"id": a.base_id, "path": str(Path(a.base_path).resolve()), "chat_kwargs": models.get(a.base_id, {}).get("chat_kwargs", {})},
        "backend": a.backend,
        "quant": a.quant,
        "active_version": None,
    }
    root.mkdir(parents=True, exist_ok=True)
    save_project(root, project)
    return {"project_dir": str(root), "project": project}


def cmd_plan(a: argparse.Namespace) -> dict:
    from engine.contract import ProjectPaths, atomic_write_json  # noqa: WPS433

    plan = _make_plan(a)
    run_dir = ProjectPaths(Path(a.project)).run_dir(plan["run_id"])
    atomic_write_json(run_dir / "plan.json", plan)
    return {"plan_path": str(run_dir / "plan.json"), "plan": plan}


def cmd_ingest(a: argparse.Namespace) -> dict:
    from engine.pipeline.ingest_and_verify import run_ingest  # noqa: WPS433

    from engine.contract import project_lock  # noqa: WPS433

    rules = json.loads(a.rules) if a.rules else []
    with project_lock(Path(a.project), "being trained or rebuilt"):
        return run_ingest(Path(a.project), a.inputs, rules)


def cmd_preview(a: argparse.Namespace) -> dict:
    from engine.pipeline.ingest_and_verify import preview  # noqa: WPS433

    return preview(Path(a.project), a.limit)


def cmd_rules(a: argparse.Namespace) -> dict:
    from engine.pipeline.ingest_and_verify import rules_preview  # noqa: WPS433

    return rules_preview(Path(a.project) if a.project else None, json.loads(a.texts))


def cmd_train(a: argparse.Namespace) -> dict:
    from engine.contract import ProjectPaths  # noqa: WPS433
    from engine.pipeline.train_and_gate import run_train  # noqa: WPS433

    root = Path(a.project)
    if a.resume:
        plan = json.loads((ProjectPaths(root).run_dir(a.resume) / "plan.json").read_text())
    elif a.plan:
        plan = json.loads(Path(a.plan).read_text())
    else:
        plan = _make_plan(a)
    from engine.contract import project_lock  # noqa: WPS433

    events.log(f"Run {plan['run_id']}: {plan['base_model']['id']} on {plan['backend']} ({plan['quant']}), LoRA r={plan['lora']['r']}", "info")
    with project_lock(root, "being trained"):
        return run_train(root, plan, resume=bool(a.resume))


def cmd_gate(a: argparse.Namespace) -> dict:
    from engine.pipeline.train_and_gate import run_gate_only  # noqa: WPS433

    return run_gate_only(Path(a.project), _make_plan(a), Path(a.adapter))


def cmd_arena(a: argparse.Namespace) -> dict:
    from engine.pipeline.arena import run_arena  # noqa: WPS433

    return run_arena(Path(a.project), a.prompt, a.n, a.max_new_tokens)


def cmd_serve(a: argparse.Namespace) -> dict:
    from engine.inference_worker import serve  # noqa: WPS433

    serve(Path(a.project), a.port, a.version)
    return {"stopped": True}


def cmd_export(a: argparse.Namespace) -> dict:
    from engine.adapters import export_inkpatch  # noqa: WPS433

    return export_inkpatch(Path(a.project), Path(a.out), a.version)


def cmd_import(a: argparse.Namespace) -> dict:
    from engine.adapters import import_inkpatch  # noqa: WPS433
    from engine.pipeline.train_and_gate import run_gate_only  # noqa: WPS433

    import shutil  # noqa: WPS433

    from engine.contract import project_lock  # noqa: WPS433

    with project_lock(Path(a.project), "being trained"):
        events.stage("import", "started", "Checking signature, SHA-256 hashes and every tensor")
        info = import_inkpatch(Path(a.project), Path(a.file), trust=a.trust)
        events.stage("import", "completed", f"{info['checks']['tensors']} tensors verified")
        result: dict[str, Any] = {"import": {k: v for k, v in info.items() if k != "manifest"}, "name": info["manifest"].get("name")}
        if not a.no_gate:
            try:
                result["gate"] = run_gate_only(Path(a.project), _make_plan(a), Path(info["adapter_dir"]))
            finally:
                # The unpacked copy has served its purpose (a promoted adapter lives under versions/).
                shutil.rmtree(Path(info["adapter_dir"]).parent, ignore_errors=True)
    return result


def cmd_smoke_model(a: argparse.Namespace) -> dict:
    from engine.smoke import build_smoke_model  # noqa: WPS433

    return build_smoke_model(Path(a.out), steps=a.steps)


def cmd_smoke_data(a: argparse.Namespace) -> dict:
    from engine.smoke import write_smoke_data  # noqa: WPS433

    return write_smoke_data(Path(a.out))


def cmd_fetch_model(a: argparse.Namespace) -> dict:
    from huggingface_hub import snapshot_download  # noqa: WPS433

    from engine.contract import load_contract  # noqa: WPS433

    model = load_contract("tiers.json")["models"][a.id]
    repo = model["mlx"] if a.backend == "mlx" else model["hf"]
    dest = Path(a.dest)
    events.stage("download", "started", f"Downloading {model['label']} ({repo})")

    class _Bar:
        def __init__(self, *args: Any, total: int | None = None, **kw: Any):
            self.total, self.n = total or 1, 0
            self.prog = events.Progress("download", self.total, 0, 100)

        def update(self, k: int = 1) -> None:
            self.n += k
            self.prog.update(self.n)

        def __enter__(self) -> "_Bar":
            return self

        def __exit__(self, *exc: Any) -> None:
            self.prog.update(self.total, force=True)

        def close(self) -> None:
            pass

        def set_description(self, *a: Any, **k: Any) -> None:
            pass

        def refresh(self, *a: Any, **k: Any) -> None:
            pass

    path = snapshot_download(repo_id=repo, local_dir=str(dest), allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.tiktoken", "tokenizer*"], tqdm_class=_Bar)
    events.stage("download", "completed")
    return {"path": path, "repo": repo, "model_id": a.id}


# ------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m engine", description="Inkport engine sidecar")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn: Callable[[argparse.Namespace], dict], help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        return sp

    def planning(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--hardware", help="HardwareReport JSON from the Rust supervisor (default: detect here)")
        sp.add_argument("--preset", default="standard", choices=["standard", "fast", "smoke"])
        sp.add_argument("--overrides", help="JSON object deep-merged into the plan")

    add("hello", cmd_hello, "report engine version and available backends")
    sp = add("detect", cmd_detect, "detect hardware from Python")
    sp.add_argument("--out")
    sp = add("create-project", cmd_create_project, "create a project folder (normally done by the app)")
    sp.add_argument("--root", required=True)
    sp.add_argument("--name", required=True)
    sp.add_argument("--template", required=True)
    sp.add_argument("--base-id", required=True)
    sp.add_argument("--base-path", required=True)
    sp.add_argument("--backend")
    sp.add_argument("--quant")
    sp = add("plan", cmd_plan, "write a training plan for this computer")
    sp.add_argument("--project", required=True)
    planning(sp)
    sp = add("ingest", cmd_ingest, "turn files into a practice exam with a hidden test set")
    sp.add_argument("--project", required=True)
    sp.add_argument("--inputs", nargs="+", required=True)
    sp.add_argument("--rules", help="JSON list of plain-English rules")
    sp = add("preview", cmd_preview, "sample practice questions (test items are never shown)")
    sp.add_argument("--project", required=True)
    sp.add_argument("--limit", type=int, default=12)
    sp = add("rules", cmd_rules, "compile plain-English rules and self-test them")
    sp.add_argument("--project")
    sp.add_argument("--texts", required=True, help="JSON list of rule strings")
    sp = add("train", cmd_train, "train, run the before-vs-after gate, promote or roll back")
    sp.add_argument("--project", required=True)
    sp.add_argument("--plan")
    sp.add_argument("--resume", help="run id to resume")
    planning(sp)
    sp = add("gate", cmd_gate, "gate an existing adapter against the current version")
    sp.add_argument("--project", required=True)
    sp.add_argument("--adapter", required=True)
    planning(sp)
    sp = add("arena", cmd_arena, "side-by-side: original vs trained")
    sp.add_argument("--project", required=True)
    sp.add_argument("--prompt")
    sp.add_argument("--n", type=int, default=4)
    sp.add_argument("--max-new-tokens", type=int, default=256)
    sp = add("serve", cmd_serve, "serve the active version for the local API gateway")
    sp.add_argument("--project", required=True)
    sp.add_argument("--port", type=int, default=0)
    sp.add_argument("--version")
    sp = add("export", cmd_export, "export a signed .inkpatch")
    sp.add_argument("--project", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--version")
    sp = add("import", cmd_import, "verify a .inkpatch and gate it before activation")
    sp.add_argument("--project", required=True)
    sp.add_argument("--file", required=True)
    sp.add_argument("--trust", action="store_true", help="trust this publisher from now on")
    sp.add_argument("--no-gate", action="store_true")
    planning(sp)
    sp = add("smoke-model", cmd_smoke_model, "build the tiny offline test model")
    sp.add_argument("--out", required=True)
    sp.add_argument("--steps", type=int, default=800)
    sp = add("smoke-data", cmd_smoke_data, "write synthetic small-business files")
    sp.add_argument("--out", required=True)
    sp = add("fetch-model", cmd_fetch_model, "download a catalog model")
    sp.add_argument("--id", required=True)
    sp.add_argument("--backend", default="torch")
    sp.add_argument("--dest", required=True)
    return p


def _classify(exc: BaseException) -> tuple[str, bool]:
    from engine.adapters import PatchRejected  # noqa: WPS433
    from engine.pipeline.train_and_gate import LeakageError  # noqa: WPS433

    from engine.contract import ProjectBusy  # noqa: WPS433

    msg = str(exc).lower()
    if isinstance(exc, ProjectBusy):
        return "BUSY", True
    if isinstance(exc, LeakageError):
        return "LEAKAGE", False
    if isinstance(exc, PatchRejected):
        return ("UNTRUSTED_PUBLISHER" if str(exc).startswith("UNTRUSTED_PUBLISHER") else "PATCH_REJECTED"), False
    if isinstance(exc, MemoryError) or exc.__class__.__name__ == "OutOfMemoryError" or any(w in msg for w in ("out of memory", "not enough memory", "unable to allocate", "cannot allocate memory", "paging file is too small")):
        return "OUT_OF_MEMORY", True
    if isinstance(exc, FloatingPointError):
        return "NAN_LOSS", True
    if isinstance(exc, FileNotFoundError):
        return "NOT_FOUND", False
    if isinstance(exc, (ValueError, KeyError)):
        return "BAD_INPUT", False
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
        return "DISK_FULL", True
    return "INTERNAL", False


def main(argv: list[str] | None = None) -> int:
    events.install_protocol_stdout()
    events.watch_parent_stdin()
    events.start_heartbeat()
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0) if exc.code in (0, None) else 2
    try:
        result = args.fn(args)
        events.done(True, **(result or {}))
        return 0
    except KeyboardInterrupt:
        events.error("CANCELLED", "Stopped by the user", retryable=True)
        events.done(False, code="CANCELLED")
        return 1
    except Exception as exc:  # noqa: BLE001
        import traceback  # noqa: WPS433

        traceback.print_exc(file=sys.stderr)
        code, retry = _classify(exc)
        message = str(exc)
        if code == "OUT_OF_MEMORY":
            on_gpu = "cuda" in message.lower() or "gpu" in message.lower() or "mps" in message.lower() or "metal" in message.lower()
            where = "The graphics card ran out of memory" if on_gpu else "Your computer ran out of memory"
            other = "programs that use the graphics card (games, video editors, browsers with many tabs)" if on_gpu else "programs (web browsers use a lot of memory)"
            message = f"{where}. Close other {other} and try again. If it keeps happening, restart the computer first, or use Quick run."
        events.error(code, message, retryable=retry)
        events.done(False, code=code, message=message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
