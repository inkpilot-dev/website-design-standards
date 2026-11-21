"""Adapter packaging: signed .inkpatch export/import, MLX->PEFT conversion, GGUF export.

A .inkpatch is a zip with:
    manifest.json      - base model fingerprint, LoRA config, eval summary, sha256 of every file
    signature.json     - Ed25519 signature over canonical JSON of manifest.json
    adapter/...        - adapter_model.safetensors + adapter_config.json (PEFT format)
    report.json        - the gate report that promoted it
Import never executes anything from the archive: it checks the zip structure, every sha256,
the signature and publisher trust, the adapter config, and every tensor's name, shape, dtype
and finiteness against the project's base model. An imported adapter is only a *candidate*:
it must still pass this machine's own hidden-test gate before it can become active.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

from engine import events
from engine.contract import ProjectPaths, active_version, canonical_json, load_project, sha256_bytes, sha256_file

MAX_ENTRY_BYTES = 4 * 1024**3
MAX_RANK = 256
ALLOWED_ENTRIES = {
    "manifest.json",
    "signature.json",
    "report.json",
    "adapter/adapter_model.safetensors",
    "adapter/adapter_config.json",
    "adapter/inkport_adapter.json",
}
ALLOWED_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
KEY_RE = re.compile(r"^base_model\.model\.(?:model\.)?(?:language_model\.)?(?:model\.)?layers\.(\d+)\.(?:self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.(lora_A|lora_B)\.weight$")
DORA_RE = re.compile(r"^base_model\.model\.(?:model\.)?(?:language_model\.)?(?:model\.)?layers\.(\d+)\.(?:self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_magnitude_vector(?:\.weight)?$")
ALLOWED_DTYPES = {"F16", "BF16", "F32"}


class PatchRejected(Exception):
    pass


def home_dir() -> Path:
    return Path(os.environ.get("INKPORT_HOME") or (Path.home() / ".inkport"))


# ------------------------------------------------------------------------------------------
# Keys and fingerprints
# ------------------------------------------------------------------------------------------


def _keys():
    from cryptography.hazmat.primitives import serialization  # noqa: WPS433
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: WPS433

    kdir = home_dir() / "keys"
    kdir.mkdir(parents=True, exist_ok=True)
    path = kdir / "publisher_ed25519.pem"
    if path.is_file():
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    else:
        key = Ed25519PrivateKey.generate()
        path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    return key, pub


def trusted_keys() -> set[str]:
    p = home_dir() / "keys" / "trusted_publishers.json"
    return set(json.loads(p.read_text())) if p.is_file() else set()


def trust_key(pub_hex: str) -> None:
    p = home_dir() / "keys" / "trusted_publishers.json"
    keys = trusted_keys() | {pub_hex}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sorted(keys), indent=2))


def base_fingerprint(model_path: str) -> dict:
    cfg_path = Path(model_path) / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    cfg = cfg.get("text_config", cfg)
    keys = ["model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size", "head_dim", "num_experts", "moe_intermediate_size"]
    shape = {k: cfg.get(k) for k in keys if k in cfg}
    return {"config": shape, "sha256": sha256_bytes(canonical_json(shape))}


def expected_dims(cfg: dict, module: str) -> tuple[int, int]:
    d = int(cfg["hidden_size"])
    heads = int(cfg.get("num_attention_heads", 1))
    kv = int(cfg.get("num_key_value_heads", heads))
    hd = int(cfg.get("head_dim") or d // heads)
    ff = int(cfg.get("intermediate_size", 4 * d))
    return {
        "q_proj": (d, heads * hd),
        "k_proj": (d, kv * hd),
        "v_proj": (d, kv * hd),
        "o_proj": (heads * hd, d),
        "gate_proj": (d, ff),
        "up_proj": (d, ff),
        "down_proj": (ff, d),
    }[module]


# ------------------------------------------------------------------------------------------
# Safetensors validation (header parsed by hand; nothing in the file is executed)
# ------------------------------------------------------------------------------------------


def read_safetensors_header(path: Path) -> dict:
    size = path.stat().st_size
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise PatchRejected("adapter weights file is truncated")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > 64 * 1024 * 1024 or 8 + n > size:
            raise PatchRejected("adapter weights header is invalid")
        header = json.loads(f.read(n).decode("utf-8"))
    data_len = size - 8 - n
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        if not (0 <= start <= end <= data_len):
            raise PatchRejected(f"tensor {name} points outside the file")
    return header


def validate_peft_adapter(adir: Path, model_path: str) -> dict:
    cfg_file = adir / "adapter_config.json"
    weights = adir / "adapter_model.safetensors"
    if not cfg_file.is_file() or not weights.is_file():
        raise PatchRejected("adapter is missing adapter_config.json or adapter_model.safetensors")
    acfg = json.loads(cfg_file.read_text())
    if acfg.get("peft_type") != "LORA":
        raise PatchRejected("only LoRA/DoRA adapters are accepted")
    if acfg.get("modules_to_save"):
        raise PatchRejected("adapters that replace whole layers (modules_to_save) are not accepted")
    if "auto_map" in acfg or acfg.get("trust_remote_code"):
        raise PatchRejected("adapters that require custom code are not accepted")
    r = int(acfg.get("r", 0))
    if not 1 <= r <= MAX_RANK:
        raise PatchRejected(f"LoRA rank {r} is outside 1..{MAX_RANK}")
    targets = acfg.get("target_modules") or []
    if isinstance(targets, str):
        targets = [targets]
    if not set(targets) <= ALLOWED_TARGETS:
        raise PatchRejected(f"unexpected target modules: {sorted(set(targets) - ALLOWED_TARGETS)}")
    base_cfg = json.loads((Path(model_path) / "config.json").read_text())
    base_cfg = base_cfg.get("text_config", base_cfg)
    n_layers = int(base_cfg["num_hidden_layers"])
    header = read_safetensors_header(weights)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        raise PatchRejected("adapter contains no weights")
    for name, info in tensors.items():
        if info["dtype"] not in ALLOWED_DTYPES:
            raise PatchRejected(f"tensor {name} has unsupported dtype {info['dtype']}")
        m = KEY_RE.match(name)
        md = DORA_RE.match(name)
        if not m and not md:
            raise PatchRejected(f"unexpected tensor name {name}")
        layer = int((m or md).group(1))
        module = (m or md).group(2)
        if layer >= n_layers:
            raise PatchRejected(f"tensor {name} targets layer {layer}, but the base model has {n_layers}")
        d_in, d_out = expected_dims(base_cfg, module)
        shape = info["shape"]
        if md:
            if shape != [d_out]:
                raise PatchRejected(f"tensor {name} has shape {shape}, expected [{d_out}]")
            continue
        expect = [r, d_in] if m.group(3) == "lora_A" else [d_out, r]
        if shape != expect:
            raise PatchRejected(f"tensor {name} has shape {shape}, expected {expect}")
    finite = None
    try:
        import torch  # noqa: WPS433
        from safetensors.torch import load_file  # noqa: WPS433

        loaded = load_file(str(weights))
        finite = all(bool(torch.isfinite(t.float()).all()) for t in loaded.values())
        max_norm = max(float(t.float().norm()) for t in loaded.values())
        if not finite:
            raise PatchRejected("adapter contains NaN or infinite values")
        if max_norm > 1e4:
            raise PatchRejected(f"adapter weights are implausibly large (norm {max_norm:.1f})")
    except ImportError:
        pass
    return {"tensors": len(tensors), "rank": r, "targets": targets, "finite": finite}


# ------------------------------------------------------------------------------------------
# Export / import
# ------------------------------------------------------------------------------------------


def export_inkpatch(root: Path, out_file: Path, version: Optional[str] = None) -> dict:
    project = load_project(root)
    paths = ProjectPaths(root)
    version = version or (active_version(root) or {}).get("version")
    if not version:
        raise PatchRejected("there is no trained version to export yet")
    vdir = paths.versions / version
    adir = vdir / "adapter"
    files = {
        "adapter/adapter_model.safetensors": adir / "adapter_model.safetensors",
        "adapter/adapter_config.json": adir / "adapter_config.json",
        "report.json": vdir / "report.json",
    }
    if (adir / "inkport_adapter.json").is_file():
        files["adapter/inkport_adapter.json"] = adir / "inkport_adapter.json"
    report = json.loads((vdir / "report.json").read_text())
    acfg = json.loads((adir / "adapter_config.json").read_text())
    key, pub = _keys()
    manifest = {
        "format": "inkpatch",
        "version": 1,
        "name": project["name"],
        "template": project["template"],
        "base_model": {"id": project["base_model"]["id"], "fingerprint": base_fingerprint(project["base_model"]["path"])["sha256"]},
        "adapter": {"r": acfg.get("r"), "alpha": acfg.get("lora_alpha"), "targets": acfg.get("target_modules"), "dora": bool(acfg.get("use_dora"))},
        "files": {name: sha256_file(p) for name, p in files.items()},
        "eval": {k: report.get(k) for k in ("decision", "label", "test", "general") if k in report},
        "created_at": int(time.time()),
        "publisher": {"public_key": pub},
    }
    sig = key.sign(canonical_json(manifest)).hex()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(out_file.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("signature.json", json.dumps({"alg": "ed25519", "public_key": pub, "signature": sig}))
        for name, p in files.items():
            z.write(p, name)
    os.replace(tmp, out_file)
    digest = sha256_file(out_file)
    events.emit("artifact", kind="inkpatch", path=str(out_file), sha256=digest)
    return {"path": str(out_file), "sha256": digest, "version": version, "publisher": pub, "files": manifest["files"]}


def import_inkpatch(root: Path, patch: Path, trust: bool = False) -> dict:
    """Validate and unpack. Returns {"adapter_dir", ...}; raises PatchRejected on any problem."""
    from cryptography.exceptions import InvalidSignature  # noqa: WPS433
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey  # noqa: WPS433

    project = load_project(root)
    if not zipfile.is_zipfile(patch):
        raise PatchRejected("this file is not a Inkport adapter package")
    with zipfile.ZipFile(patch) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        for i in infos:
            n = i.filename
            if n.startswith(("/", "\\")) or ".." in Path(n).parts or ":" in n or "\\" in n:
                raise PatchRejected(f"unsafe path in package: {n}")
            if (i.external_attr >> 16) & 0o170000 == 0o120000:
                raise PatchRejected(f"symbolic links are not allowed: {n}")
            if i.is_dir():
                continue
            if n not in ALLOWED_ENTRIES:
                raise PatchRejected(f"unexpected file in package: {n}")
            if i.file_size > MAX_ENTRY_BYTES or (i.compress_size and i.file_size / max(1, i.compress_size) > 200):
                raise PatchRejected(f"file too large or suspiciously compressed: {n}")
        if len(names) != len(set(names)):
            raise PatchRejected("duplicate file names in package")
        for required in ("manifest.json", "signature.json", "adapter/adapter_model.safetensors", "adapter/adapter_config.json"):
            if required not in names:
                raise PatchRejected(f"package is missing {required}")
        manifest = json.loads(z.read("manifest.json"))
        sig = json.loads(z.read("signature.json"))
        if manifest.get("format") != "inkpatch" or manifest.get("version") != 1:
            raise PatchRejected("unsupported package format")
        if sig.get("alg") != "ed25519" or sig.get("public_key") != manifest.get("publisher", {}).get("public_key"):
            raise PatchRejected("signature does not match the publisher in the manifest")
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(sig["public_key"])).verify(bytes.fromhex(sig["signature"]), canonical_json(manifest))
        except (InvalidSignature, ValueError) as exc:
            raise PatchRejected("signature verification failed: the package was modified or is not genuine") from exc
        listed = manifest.get("files", {})
        for name in names:
            if name in ("manifest.json", "signature.json") or name.endswith("/"):
                continue
            if name not in listed:
                raise PatchRejected(f"{name} is not listed in the signed manifest")
            if sha256_bytes(z.read(name)) != listed[name]:
                raise PatchRejected(f"SHA-256 mismatch for {name}: the file was altered")
        for name in listed:
            if name not in names:
                raise PatchRejected(f"signed file {name} is missing from the package")
        _, local_pub = _keys()
        pub = sig["public_key"]
        trusted = pub == local_pub or pub in trusted_keys()
        if not trusted and not trust:
            raise PatchRejected(f"UNTRUSTED_PUBLISHER:{pub}")
        if trust and not trusted:
            trust_key(pub)
        fp = base_fingerprint(project["base_model"]["path"])["sha256"]
        if manifest["base_model"]["fingerprint"] != fp:
            raise PatchRejected(f"this adapter was trained for a different base model ({manifest['base_model']['id']})")
        dest = ProjectPaths(root).root / "imports" / (manifest["files"]["adapter/adapter_model.safetensors"][:16])
        if dest.exists():
            shutil.rmtree(dest)
        adir = dest / "adapter"
        adir.mkdir(parents=True)
        for name in names:
            if name.startswith("adapter/") and not name.endswith("/"):
                (adir / Path(name).name).write_bytes(z.read(name))
    checks = validate_peft_adapter(adir, project["base_model"]["path"])
    return {"adapter_dir": str(adir), "publisher": pub, "trusted": True, "manifest": manifest, "checks": checks, "sha256_verified": True, "signature_verified": True}


# ------------------------------------------------------------------------------------------
# Format conversion
# ------------------------------------------------------------------------------------------


def mlx_to_peft(mlx_dir: Path, out_dir: Path, plan: dict) -> None:
    """Convert mlx-lm LoRA weights to PEFT: transpose A/B and map scale -> alpha = scale * r."""
    import numpy as np  # noqa: WPS433
    from safetensors.numpy import save_file  # noqa: WPS433

    import mlx.core as mx  # noqa: WPS433

    cfg = json.loads((mlx_dir / "adapter_config.json").read_text())
    lp = cfg["lora_parameters"]
    r = int(lp["rank"])
    weights = mx.load(str(mlx_dir / "adapters.safetensors"))
    out: dict[str, Any] = {}
    targets = set()
    for name, arr in weights.items():
        m = re.match(r"^(?:model\.)?(?:language_model\.)?(?:model\.)?layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+)\.(lora_a|lora_b|m)$", name)
        if not m:
            continue
        layer, block, module, kind = m.groups()
        targets.add(module)
        prefix = f"base_model.model.model.layers.{layer}.{block}.{module}"
        a = np.array(arr.astype(mx.float32))
        if kind == "lora_a":
            out[f"{prefix}.lora_A.weight"] = a.T.copy()
        elif kind == "lora_b":
            out[f"{prefix}.lora_B.weight"] = a.T.copy()
        else:
            out[f"{prefix}.lora_magnitude_vector"] = a.copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "adapter_model.safetensors"))
    peft_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": r,
        "lora_alpha": float(lp["scale"]) * r,
        "lora_dropout": float(lp.get("dropout", 0.0)),
        "target_modules": sorted(targets),
        "use_dora": cfg.get("fine_tune_type") == "dora",
        "bias": "none",
        "base_model_name_or_path": plan["base_model"]["path"],
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(peft_cfg, indent=2))


def try_export_gguf(plan: dict, adapter_dir: Path, out_dir: Path) -> Optional[Path]:
    """GGUF LoRA via the bundled llama.cpp converter (INKPORT_LLAMA_CPP = <bundle>/llama/convert)."""
    llama = os.environ.get("INKPORT_LLAMA_CPP")
    if not llama:
        return None
    script = Path(llama) / "convert_lora_to_gguf.py"
    if not script.is_file():
        events.log(f"GGUF export skipped: {script} not found", "warn")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "adapter-f16.gguf"
    env = {**os.environ, "PYTHONPATH": str(Path(llama)) + os.pathsep + os.environ.get("PYTHONPATH", ""), "NO_LOCAL_GGUF": "1"}
    env.pop("INKPORT_WATCH_STDIN", None)
    cmd = [sys.executable, "-X", "utf8", str(script), "--base", plan["base_model"]["path"], "--outfile", str(out), "--outtype", "f16", str(adapter_dir)]
    try:
        # stdin=DEVNULL: the child must never inherit our stdin pipe (on Windows that can hang it
        # at start-up). No console window on Windows; bounded time so a stuck export cannot hold
        # a finished training run hostage (the version is then served by the Python worker).
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=1800, creationflags=0x08000000 if os.name == "nt" else 0)
    except subprocess.TimeoutExpired:
        events.log("GGUF export timed out; this version will be served without the fast answer engine", "warn")
        return None
    except OSError as exc:
        events.log(f"GGUF export could not start: {exc}", "warn")
        return None
    if proc.returncode != 0 or not out.is_file():
        events.log("GGUF export failed: " + proc.stderr[-400:], "warn")
        return None
    return out
