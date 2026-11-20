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
