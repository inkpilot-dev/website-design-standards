"""Bundled-starter specifics: GGUF bases must reach the training plan intact."""

from engine.hardware_profile import make_plan, select_tier


def _report(vendor: str = "cpu") -> dict:
    gpus = [] if vendor == "cpu" else [{"vendor": vendor, "vram_gb": 12.0, "unified": False}]
    r = {"gpus": gpus, "ram_gb": 16, "apple_silicon": False}
    r["recommendation"] = select_tier(r)
    return r


def test_gguf_base_model_fields_reach_the_plan(tmp_path):
    project = {"base_model": {"id": "qwen3-1.7b", "path": str(tmp_path), "gguf": "qwen3-1.7b-q4_k_m.gguf", "format": "gguf", "bundled": True, "chat_kwargs": {"enable_thinking": False}}, "backend": "torch"}
    plan = make_plan(project, _report("cpu"), preset="fast")
    assert plan["base_model"]["gguf"] == "qwen3-1.7b-q4_k_m.gguf"
    assert plan["base_model"]["chat_kwargs"] == {"enable_thinking": False}
    assert plan["quant"] == "fp32" and plan["backend"] == "torch"


def test_gguf_base_on_nvidia_trains_in_bf16_without_bitsandbytes(tmp_path):
    project = {"base_model": {"id": "q", "path": str(tmp_path), "gguf": "q.gguf"}, "backend": "unsloth"}
    plan = make_plan(project, _report("nvidia"), preset="fast")
    assert plan["quant"] == "bf16" and plan["backend"] == "torch"


def test_gguf_export_never_inherits_stdin(tmp_path, monkeypatch):
    """Windows: a child that inherits the engine's stdin pipe hangs at start-up (the engine polls
    that pipe to notice a dead supervisor), so the converter must get its own stdin and a time limit."""
    import subprocess

    from engine import adapters

    conv = tmp_path / "convert"
    conv.mkdir()
    (conv / "convert_lora_to_gguf.py").write_text("")
    monkeypatch.setenv("INKPORT_LLAMA_CPP", str(conv))
    monkeypatch.setenv("INKPORT_WATCH_STDIN", "1")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs, cmd=cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    monkeypatch.setattr(adapters.subprocess, "run", fake_run)
    assert adapters.try_export_gguf({"base_model": {"path": str(tmp_path)}}, tmp_path / "adapter", tmp_path / "out") is None
    assert seen["stdin"] is subprocess.DEVNULL
    assert seen["timeout"] and seen["timeout"] <= 3600
    assert "INKPORT_WATCH_STDIN" not in seen["env"]
    assert seen["cmd"][1:3] == ["-X", "utf8"]


def test_format_check_needs_a_real_drop():
    """Two malformed answers out of ~90 must not veto a model (it used to: a fixed 2-point limit)."""
    from engine.pipeline.train_and_gate import format_check

    ref = [True] * 88
    assert format_check(1.0, 85 / 88, ref, [True] * 85 + [False] * 3)  # 3.4 points, p = 0.25: noise
    assert not format_check(1.0, 80 / 88, ref, [True] * 80 + [False] * 8)  # 9 points, p = 0.008: real
    assert format_check(1.0, 0.99, ref, [True] * 88)  # within the 2-point tolerance
    # A malformed answer is already a wrong answer in the task score: a model that gets many more
    # questions right is not refused for the few it left malformed (95% against 58% once was).
    ref96 = [True] * 96
    assert format_check(1.0, 91 / 96, ref96, [True] * 91 + [False] * 5, task_gain=35)
    assert not format_check(1.0, 91 / 96, ref96, [True] * 91 + [False] * 5, task_gain=6)  # gains little, breaks more
    assert not format_check(1.0, 91 / 96, ref96, [True] * 91 + [False] * 5)  # gain unknown: the plain rule
    assert format_check(None, 0.5, None, [])  # nothing to compare
    assert not format_check(1.0, 0.9, None, [True] * 88)  # old cached baseline: plain threshold


def test_a_saved_mac_adapter_names_the_modules_it_adapts():
    """Whoever loads the adapter must adapt exactly the modules it has weights for."""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "engine" / "backends" / "mlx_backend.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "adapted_modules")
    scope: dict = {"Any": object}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "mlx_backend", "exec"), scope)  # the module itself needs MLX to import
    names = ["model.layers.0.self_attn.q_proj.lora_a", "model.layers.0.self_attn.q_proj.lora_b", "model.layers.27.mlp.down_proj.lora_b", "language_model.model.layers.3.mlp.up_proj.m", "model.embed_tokens.weight"]
    assert scope["adapted_modules"](names) == ["mlp.down_proj", "mlp.up_proj", "self_attn.q_proj"]
    assert scope["adapted_modules"]([]) == []
