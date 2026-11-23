"""Fold the LoRA adapter into the base model and write an Ollama Modelfile.

    python -m paper_distill.export

Produces out/paper-merged (a normal model folder) and out/Modelfile. Converting the
merged folder to GGUF for llama.cpp or Ollama is two llama.cpp commands; see the README.
"""
from __future__ import annotations

import argparse

from .config import load


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--base-model", default=None)
    a = ap.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load(a.config)
    adapter = cfg["root"] / cfg["train"]["output_dir"]
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"No adapter in {adapter}. Run `python -m paper_distill.train` first.")
    base = a.base_model or cfg["student"]["base_model"]
    merged = cfg["root"] / "out" / "paper-merged"
    # Merge on the CPU at the precision the base was published in (bf16 for Qwen3-4B).
    # Merging into the 4-bit weights used for training would lose accuracy.
    model = AutoModelForCausalLM.from_pretrained(base, dtype="auto")
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(merged)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(merged)
    inf = cfg["inference"]
    system = cfg["system_prompt"].replace('"""', "'''")
    (cfg["root"] / "out" / "Modelfile").write_text(
        "FROM ./paper-Q4_K_M.gguf\n"
        f'SYSTEM """{system}"""\n'
        f"PARAMETER temperature {inf['temperature']}\n"
        f"PARAMETER top_p {inf['top_p']}\n"
        "PARAMETER num_ctx 8192\n", encoding="utf-8")
    print(f"export: merged model written to {merged}")
    print(f"export: Modelfile written to {cfg['root'] / 'out' / 'Modelfile'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
