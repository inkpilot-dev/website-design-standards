"""Unified hardware router: maps a HardwareReport to a tier and a concrete TrainingPlan.

The tier table is contracts/tiers.json (shared with the Rust supervisor, which shows the
recommendation before Python is even installed). This module adds what needs Python:
reading the model's config.json, the memory model from the implementation plan, automatic
downshifting when the plan would not fit, and the full plan.json for `engine train`.
"""

from __future__ import annotations

import json
import os
import platform
import uuid
from pathlib import Path
from typing import Any, Optional

from engine.contract import load_contract

GIB = 1024**3

# ------------------------------------------------------------------------------------------
# Tier selection (mirrors src-tauri/src/hardware.rs::select_tier; both are tested against
# contracts/fixtures/tier_cases.json)
# ------------------------------------------------------------------------------------------


def primary_accelerator(report: dict) -> tuple[str, float]:
    if report.get("apple_silicon"):
        return "apple", float(report.get("ram_gb", 0.0))
    best: tuple[str, float] | None = None
    order = {"nvidia": 0, "amd": 1, "intel": 2}
    for g in report.get("gpus", []):
        vendor = g.get("vendor")
        if vendor not in order:
            continue
        vram = float(g.get("vram_gb", 0.0))
        if best is None or vram > best[1] or (vram == best[1] and order[vendor] < order[best[0]]):
            best = (vendor, vram)
    return best if best else ("cpu", 0.0)


def select_tier(report: dict) -> dict:
    tiers = load_contract("tiers.json")
    vendor, mem = primary_accelerator(report)
    chosen = None
    for t in tiers["tiers"]:
        if t["vendor"] == vendor and t["min_gb"] <= mem < t["max_gb"]:
            chosen = t
            break
    if chosen is None:
        chosen = next(t for t in tiers["tiers"] if t["id"] == "X")
    model = tiers["models"][chosen["model"]]
    return {
        "tier": chosen["id"],
        "vendor": vendor,
        "memory_gb": round(mem, 2),
        "model_id": chosen["model"],
        "model_label": model["label"],
        "backend": chosen["backend"],
        "quant": chosen["quant"],
        "lora_r": chosen["lora_r"],
        "micro_batch": chosen["micro_batch"],
        "grad_accum": chosen["grad_accum"],
        "max_seq": chosen["max_seq"],
        "stage2": chosen["stage2"],
        "rft_k": chosen["rft_k"],
        "plain": chosen["plain"],
    }


# ------------------------------------------------------------------------------------------
# Python-side detection (used when the engine runs without the Rust supervisor, e.g. CLI)
# ------------------------------------------------------------------------------------------


def _ram_gb() -> float:
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB
    except (ValueError, OSError):
        pass
    if platform.system() == "Windows":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        return stat.ullTotalPhys / GIB
    return 0.0


def detect() -> dict:
    system = platform.system()
    machine = platform.machine().lower()
    apple = system == "Darwin" and machine in ("arm64", "aarch64")
    gpus: list[dict] = []
    try:
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                is_rocm = getattr(torch.version, "hip", None) is not None
                gpus.append(
                    {
                        "vendor": "amd" if is_rocm else "nvidia",
                        "name": props.name,
                        "vram_gb": round(total / GIB, 2),
                        "vram_free_gb": round(free / GIB, 2),
                        "driver": None,
                        "compute_capability": f"{props.major}.{props.minor}",
                        "unified": False,
                    }
                )
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            props = torch.xpu.get_device_properties(0)
            gpus.append({"vendor": "intel", "name": props.name, "vram_gb": round(props.total_memory / GIB, 2), "vram_free_gb": None, "driver": None, "compute_capability": None, "unified": False})
    except Exception:  # noqa: BLE001
        pass
    ram = _ram_gb()
    if apple:
        gpus.append({"vendor": "apple", "name": "Apple Silicon GPU", "vram_gb": round(ram, 2), "vram_free_gb": None, "driver": None, "compute_capability": None, "unified": True})
    report = {
        "os": {"Darwin": "macos", "Windows": "windows", "Linux": "linux"}.get(system, "other"),
        "arch": machine,
        "cpu_brand": platform.processor() or machine,
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": round(ram, 2),
        "disk_free_gb": 0.0,
        "gpus": gpus,
        "apple_silicon": apple,
    }
    report["recommendation"] = select_tier(report)
    return report


def apple_working_set_gb(ram_gb: float) -> float:
    """Metal's recommendedMaxWorkingSetSize; refined from MLX when it is installed."""
    try:
        import mlx.core as mx  # noqa: WPS433

        info = mx.metal.device_info()
        return float(info["max_recommended_working_set_size"]) / GIB
    except Exception:  # noqa: BLE001
        return ram_gb * (0.67 if ram_gb <= 36 else 0.75)


# ------------------------------------------------------------------------------------------
# Memory model (implementation plan section 1.5)
# ------------------------------------------------------------------------------------------

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_model_config(model_path: str) -> Optional[dict]:
    p = Path(model_path) / "config.json"
    if p.is_file():
        cfg = json.loads(p.read_text(encoding="utf-8"))
        return cfg.get("text_config", cfg)
    return None


def estimate_memory_gb(cfg: dict, quant: str, r: int, micro_batch: int, seq: int, backend: str) -> dict:
    d = int(cfg.get("hidden_size", 2048))
    layers = int(cfg.get("num_hidden_layers", 24))
    vocab = int(cfg.get("vocab_size", 32000))
    heads = int(cfg.get("num_attention_heads", 16))
    kv = int(cfg.get("num_key_value_heads", heads))
    head_dim = int(cfg.get("head_dim") or d // heads)
    n_experts = int(cfg.get("num_experts", 0) or 0)
    top_k = int(cfg.get("num_experts_per_tok", 0) or 0)
    d_ff = int(cfg.get("moe_intermediate_size") or cfg.get("intermediate_size", 4 * d)) if n_experts else int(cfg.get("intermediate_size", 4 * d))
    tied = bool(cfg.get("tie_word_embeddings", False))
    q_out, kv_out = heads * head_dim, kv * head_dim

    attn = d * q_out + 2 * d * kv_out + q_out * d
    mlp_one = 3 * d * d_ff
    mlp = mlp_one * (n_experts if n_experts else 1)
    n_emb = vocab * d * (1 if tied else 2)
    n_total = layers * (attn + mlp) + n_emb

    b_q = {"nf4": 0.52, "mlx4": 0.56, "bf16": 2.0, "fp16": 2.0, "fp32": 4.0}[quant]
    b_e = 0.56 if quant == "mlx4" else (4.0 if quant == "fp32" else 2.0)
    m_w = (n_total - n_emb) * b_q + n_emb * b_e

    attn_lora = r * ((d + q_out) + 2 * (d + kv_out) + (q_out + d))
    if n_experts:
        mlp_lora = max(1, r // 8) * 3 * (d + d_ff) * n_experts
    else:
        mlp_lora = r * 3 * (d + d_ff)
    p_lora = layers * (attn_lora + mlp_lora)
    m_lora = 16 * p_lora

    d_ff_active = d_ff * (top_k if n_experts else 1)
    act_bytes = 4 if quant == "fp32" else 2
    m_act = micro_batch * seq * layers * d * act_bytes * 2 + micro_batch * seq * (4 * d + 3 * d_ff_active) * act_bytes * 2 + micro_batch * min(seq, 1024) * vocab * 4
    m_ovh = 0.5 * GIB if backend == "mlx" else 0.8 * GIB
    total = m_w + m_lora + m_act + m_ovh
    return {
        "weights_gb": m_w / GIB,
        "lora_gb": m_lora / GIB,
        "activations_gb": m_act / GIB,
        "overhead_gb": m_ovh / GIB,
        "total_gb": total / GIB,
        "params_b": n_total / 1e9,
        "lora_params_m": p_lora / 1e6,
    }


def fit_config(cfg: dict, rec: dict, budget_gb: float) -> tuple[dict, dict]:
    """Downshift micro_batch, then seq, then rank until the estimate fits budget_gb."""
    mb, seq, r = rec["micro_batch"], rec["max_seq"], rec["lora_r"]
    ga = rec["grad_accum"]
    for _ in range(12):
        est = estimate_memory_gb(cfg, rec["quant"], r, mb, seq, rec["backend"])
        if est["total_gb"] <= budget_gb:
            return {"micro_batch": mb, "grad_accum": ga, "max_seq": seq, "lora_r": r}, est
        if mb > 1:
            mb //= 2
            ga *= 2
        elif seq > 1024:
            seq //= 2
        elif r > 8:
            r //= 2
        else:
            break
    est = estimate_memory_gb(cfg, rec["quant"], r, mb, seq, rec["backend"])
    return {"micro_batch": mb, "grad_accum": ga, "max_seq": seq, "lora_r": r}, est


def memory_budget_gb(report: dict, rec: dict) -> float:
    if rec["vendor"] == "apple":
        return 0.80 * apple_working_set_gb(float(report.get("ram_gb", 16)))
    for g in report.get("gpus", []):
        if g.get("vendor") == rec["vendor"]:
            free = g.get("vram_free_gb") or g.get("vram_gb", 0)
            return 0.88 * float(free)
    return 0.5 * float(report.get("ram_gb", 8))  # CPU training uses system RAM


# ------------------------------------------------------------------------------------------
# Plan construction
# ------------------------------------------------------------------------------------------

PRESETS = {
    # A real run: a few epochs over the user's examples, then outcome-filtered refinement.
    "standard": {"budget_minutes": 480},
    "fast": {"budget_minutes": 90},
    # Synthetic smoke test with the tiny local model (scripts/verify_e2e.sh).
    "smoke": {"budget_minutes": 15},
}


def make_plan(project: dict, report: dict, preset: str = "standard", overrides: Optional[dict] = None) -> dict:
    rec = dict(report.get("recommendation") or select_tier(report))
    backend = project.get("backend") or rec["backend"]
    fmt = project.get("base_model", {}).get("format")
    if fmt == "mlx" and report.get("apple_silicon"):
        backend = "mlx"  # MLX weights can only be trained by MLX
    if backend == "unsloth":
        try:
            import unsloth  # noqa: F401,WPS433
        except Exception:  # noqa: BLE001
            backend = "torch"  # fall back to plain transformers + peft (+ bitsandbytes for nf4)
    if backend == "mlx" and not report.get("apple_silicon"):
        backend = "torch"
    rec["backend"] = backend

    model_path = project["base_model"]["path"]
    cfg = read_model_config(model_path)
    fitted, est = ({"micro_batch": rec["micro_batch"], "grad_accum": rec["grad_accum"], "max_seq": rec["max_seq"], "lora_r": rec["lora_r"]}, None)
    if cfg is not None and preset != "smoke":
        fitted, est = fit_config(cfg, rec, memory_budget_gb(report, rec))

    quant = rec["quant"]
    if project["base_model"].get("gguf"):
        # GGUF bases are dequantized at load: fp32 on CPU, bf16 on a GPU (bitsandbytes cannot
        # re-quantize them, and they are already compact on disk).
        quant = "bf16" if rec["vendor"] in ("nvidia", "amd", "intel") and backend in ("torch", "unsloth") else "fp32"
        backend = "torch" if backend == "unsloth" else backend
        rec["backend"] = backend
    if backend == "torch" and quant == "nf4":
        try:
            import bitsandbytes  # noqa: F401,WPS433
        except Exception:  # noqa: BLE001
            quant = "bf16"
    if project.get("quant"):
        quant = project["quant"]

    r = fitted["lora_r"]
    plan: dict[str, Any] = {
        "run_id": "run-" + uuid.uuid4().hex[:10],
        "backend": backend,
        "device": None,
        "base_model": {
            **{k: v for k, v in project["base_model"].items() if k in ("gguf", "format", "label", "bundled", "pack")},
            "id": project["base_model"]["id"],
            "path": model_path,
            "chat_kwargs": project["base_model"].get("chat_kwargs", {}),
        },
        "quant": quant,
        "lora": {"r": r, "alpha": 2 * r, "dropout": 0.0, "dora": False, "targets": LORA_TARGETS},
        "sft": {
            "lr": 2e-4,
            "epochs": 3,
            "micro_batch": fitted["micro_batch"],
            "grad_accum": fitted["grad_accum"],
            "max_seq": fitted["max_seq"],
            "warmup_frac": 0.03,
            "min_lr_ratio": 0.1,
            "max_grad_norm": 1.0,
            "replay_frac": 0.10,
            "eval_every": 0,  # 0 = auto: max(20, steps/10)
            "early_stop_patience": 3,
            "checkpoint_every": 50,
            "dnll_limit": 0.15,
            "gradient_checkpointing": backend != "mlx" and quant != "fp32",
            "max_steps": None,
        },
        "stage2": {
            "mode": rec["stage2"] if not (rec["stage2"] == "grpo" and backend == "mlx") else "rft",
            "k": rec["rft_k"],
            "rounds": 2,
            "temperature": 0.8,
            "lr": 1e-4,
            "min_gain_pts": 1.0,
            "placebo": False,
        },
        "gate": {
            "min_delta_pts": 2.0,
            "max_general_drop_pts": 5.0,
            "min_test_items": 30,
            "verified_min_items": 100,
            "max_new_tokens": 384,
            "general_suite_n": 60,
        },
        "budget_minutes": PRESETS.get(preset, PRESETS["standard"])["budget_minutes"],
        "dev_eval_n": 32,
        "seed": 1234,
        "memory_estimate": est,
        "tier": "smoke" if preset == "smoke" else rec["tier"],
    }
    if preset == "smoke":
        plan["lora"].update({"r": 64, "alpha": 128})
        plan["sft"].update({"lr": 1.5e-3, "epochs": 40, "micro_batch": 16, "grad_accum": 1, "max_seq": 256, "warmup_frac": 0.05, "eval_every": 60, "early_stop_patience": 4, "checkpoint_every": 60, "replay_frac": 0.2, "gradient_checkpointing": False, "dnll_limit": 5.0})
        plan["stage2"].update({"mode": "rft", "k": 2, "rounds": 1, "lr": 1e-3})
        plan["gate"].update({"max_new_tokens": 40, "general_suite_n": 20, "min_test_items": 30})
        plan["dev_eval_n"] = 16
    if overrides:
        _deep_update(plan, overrides)
    return plan


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
