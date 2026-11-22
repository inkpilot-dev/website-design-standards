"""Produce the bundled starter model in payload/models/starter/.

Source of truth: the official safetensors release on Hugging Face (lock.json -> starter_model),
resolved to an exact commit, every file verified (LFS files by SHA-256, small files by git blob
SHA-1, both as published by the Hub API). We quantize it ourselves so no third-party
quantization has to be trusted:

  Windows / Linux: GGUF Q4_K_M (~1.1 GB for 1.7B). One file serves both jobs: llama.cpp runs
                   it for fast local inference (GPU via Vulkan/CUDA, else CPU) and PyTorch
                   dequantizes it in memory as the base for LoRA training.
  macOS:           MLX 4-bit (group size 64): trained and served by MLX on the Apple GPU.

`--smoke` swaps in the engine's 1.4M-parameter offline test model (CI / clean-machine tests).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from common import CACHE, LOCK, ROOT, die, download, git_blob_sha1, host_platform, log, payload_python, rmtree, run

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
COPY_ALONG = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "chat_template.jinja", "LICENSE")


def hf_get(path: str) -> object:
    headers = {"User-Agent": "inkport-build/1.0"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    with urllib.request.urlopen(urllib.request.Request(f"{HF}{path}", headers=headers), timeout=60) as r:
        return json.load(r)


def fetch_hf_snapshot(repo: str, revision: str, patterns: list[str]) -> tuple[Path, str]:
    commit = hf_get(f"/api/models/{repo}/revision/{revision}")["sha"]
    tree = hf_get(f"/api/models/{repo}/tree/{commit}?recursive=true")
    files = [e for e in tree if e.get("type") == "file" and any(fnmatch.fnmatch(e["path"], p) for p in patterns)]
    if not any(f["path"].endswith(".safetensors") for f in files):
        die(f"{repo}@{commit}: no safetensors weights found")
    dest = CACHE / "hf" / repo.replace("/", "--") / commit
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else None
    for f in files:
        target = dest / f["path"]
        url = f"{HF}/{repo}/resolve/{commit}/{f['path']}"
        lfs = f.get("lfs")
        if lfs:
            download(url, target, sha256=lfs["oid"], size=lfs["size"], headers=headers)
        else:
            download(url, target, headers=headers)
            if git_blob_sha1(target) != f["oid"]:
                target.unlink()
                die(f"{f['path']}: git blob SHA-1 mismatch")
        log(f"  verified {f['path']}")
    return dest, commit


def convert_gguf(py: Path, llama: Path, tools: Path, hf_dir: Path, out_file: Path, quant: str, smoke: bool) -> None:
    f16 = CACHE / f"{out_file.stem}-f16.gguf"
    conv = llama / "convert" / "convert_hf_to_gguf.py"
    env = {"PYTHONPATH": str(llama / "convert"), "NO_LOCAL_GGUF": "1"}
    if smoke:
        # The offline test tokenizer is a plain GPT-2 style byte-level BPE whose hash is not in
        # llama.cpp's list of known pre-tokenizers; declare it explicitly.
        shim = (
            "import sys, runpy; sys.path.insert(0, sys.argv[1]); import conversion.base as b; "
            "b.TextModel.get_vocab_base_pre = lambda self, tok: 'gpt-2'; "
            "sys.argv = [sys.argv[2]] + sys.argv[3:]; runpy.run_path(sys.argv[0], run_name='__main__')"
        )
        run([py, "-c", shim, llama / "convert", conv, hf_dir, "--outfile", f16, "--outtype", "f16"], env=env)
    else:
        run([py, conv, hf_dir, "--outfile", f16, "--outtype", "f16"], env=env)
    exe = ".exe" if os.name == "nt" else ""
    run([tools / f"llama-quantize{exe}", f16, out_file, quant], env={"LD_LIBRARY_PATH": str(tools)})
    f16.unlink()


# The same conversion with the processor chosen before anything is computed.
_MLX_ON_CPU = (
    "import sys\n"
    "import mlx.core as mx\n"
    "try:\n"
    "    mx.set_default_device(mx.cpu)\n"
    "except Exception as exc:\n"
    "    print('[inkport-build] could not select the processor for MLX:', exc)\n"
    "from mlx_lm import cli\n"
    "sys.argv = ['mlx_lm'] + sys.argv[1:]\n"
    "cli.main()\n"
)


def convert_mlx(args: list, target: Path, attempts: int = 3) -> None:
    """Quantise the model for MLX, trying again when the build machine's graphics card gives up.

    Hosted macOS build machines have a virtual GPU that now and then times out in the middle of
    the conversion ("[METAL] Command buffer execution failed: Caused GPU Timeout Error"). The
    conversion itself is deterministic, so it is simply run again; the last attempt runs on the
    processor, which is slower but cannot time out that way."""
    cmd = [str(a) for a in args]
    for attempt in range(1, attempts + 1):
        rmtree(target)
        last = attempt == attempts and attempts > 1
        run_cmd = [cmd[0], "-c", _MLX_ON_CPU, *cmd[3:]] if last else cmd
        log("$ " + " ".join(cmd) + ("   (on the processor)" if last else ""))
        proc = subprocess.run(run_cmd, env={**os.environ})
        if proc.returncode == 0 and (target / "config.json").is_file():
            return
        if attempt < attempts:
            log(f"model conversion failed (attempt {attempt} of {attempts}, exit code {proc.returncode}); trying again in 30 s")
            time.sleep(30)
    die(f"model conversion failed {attempts} times: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=host_platform())
    ap.add_argument("--payload", required=True)
    ap.add_argument("--smoke", action="store_true", help="bundle the tiny offline test model instead of the real starter")
    a = ap.parse_args()
    plat = a.platform
    payload = Path(a.payload)
    py = payload_python(payload, plat)
    llama = payload / "llama"
    tools = ROOT / "build" / "tools"
    spec = dict(LOCK["starter_model"])
    out = payload / "models" / "starter"
    rmtree(out)
    out.mkdir(parents=True)

    if a.smoke:
        hf_dir = CACHE / "smoke-model"
        if not (hf_dir / "config.json").is_file():
            gen_py = py if plat != "macos-aarch64" else Path(sys.executable)
            run([gen_py, "-m", "engine", "smoke-model", "--out", hf_dir], env={"PYTHONPATH": str(ROOT)})
        commit = "local-smoke"
        spec.update(id="smoke-tiny", label="Inkport offline test model (1.4M parameters)", license="Apache-2.0", params_b=0.0014, gguf_quant="Q8_0", chat_kwargs={})
    else:
        log(f"fetching {spec['hf_repo']}@{spec['revision']} from {HF}")
        hf_dir, commit = fetch_hf_snapshot(spec["hf_repo"], spec["revision"], spec["hf_files"])

    meta = {"id": spec["id"], "label": spec["label"], "license": spec["license"], "source": spec.get("hf_repo", "local"), "revision": commit, "params_b": spec["params_b"], "chat_kwargs": spec["chat_kwargs"]}
    if plat == "macos-aarch64":
        args = [py, "-m", "mlx_lm", "convert", "--hf-path", hf_dir, "--mlx-path", out / "mlx", "-q", "--q-bits", str(spec["mlx_q_bits"]), "--q-group-size", str(spec["mlx_q_group_size"])]
        rmtree(out / "mlx")
        out.rmdir()
        convert_mlx(args, out / "mlx")
        shutil.move(str(out / "mlx"), str(out.with_name("starter-mlx")))
        rmtree(out)
        shutil.move(str(out.with_name("starter-mlx")), str(out))
        meta.update(format="mlx", backend="mlx", quant="mlx4")
    else:
        gguf_name = f"{spec['id']}-{spec['gguf_quant'].lower()}.gguf"
        convert_gguf(py, llama, tools, hf_dir, out / gguf_name, spec["gguf_quant"], a.smoke)
        for name in COPY_ALONG:  # tokenizer + config: used by training and by GGUF LoRA export
            if (hf_dir / name).is_file():
                shutil.copy2(hf_dir / name, out / name)
        meta.update(format="gguf", gguf=gguf_name, backend="torch", quant="fp32")
    (out / "model.json").write_text(json.dumps(meta, indent=2))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    log(f"starter model {meta['id']} ({meta['format']}) ready: {size / 1e6:.0f} MB")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
