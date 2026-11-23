"""Fine-tune the student on the distilled dataset with LoRA.

    python -m paper_distill.train                 # full run
    python -m paper_distill.train --resume        # continue an interrupted run

The loop is written out by hand rather than built on a trainer class, so that it does
exactly three things that matter here: the loss is taken only on the teacher's text,
batches are sized by token count so long examples cannot exhaust memory, and the run
can be stopped and resumed.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from .chat import encode_example, end_of_turn
from .config import load, read_jsonl

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_base(name: str, load_4bit: bool, device: str):
    import torch
    from transformers import AutoModelForCausalLM

    if load_4bit:
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
        return AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map={"": 0})
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)


def prepare_for_training(model, load_4bit: bool):
    """Switch on activation checkpointing (trades speed for memory) in a way the adapters can train through."""
    if load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # transformers 5 renamed this; fall back to the old name on earlier releases
        enable = getattr(model, "activation_checkpointing_enable", None) or model.gradient_checkpointing_enable
        enable()
        model.enable_input_require_grads()  # otherwise checkpointing cuts the graph before the adapters
    model.config.use_cache = False
    return model


def add_lora(model, t: dict, load_4bit: bool):
    import torch
    from peft import LoraConfig, get_peft_model

    model = prepare_for_training(model, load_4bit)
    linear = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    targets = LORA_TARGETS if set(LORA_TARGETS) <= linear else "all-linear"
    cfg = LoraConfig(r=t["lora_r"], lora_alpha=t["lora_alpha"], lora_dropout=t["lora_dropout"],
                     target_modules=targets, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def make_batches(lengths: list[int], budget: int, rng: random.Random) -> list[list[int]]:
    """Group examples of similar length so that padded size per batch stays within the token budget."""
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    batches = []
    for start in range(0, len(idx), 256):
        chunk = sorted(idx[start:start + 256], key=lambda i: lengths[i])
        cur: list[int] = []
        for i in chunk:  # ascending, so lengths[i] is the longest in the batch so far
            if cur and (len(cur) + 1) * lengths[i] > budget:
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(examples: list[dict], pad_id: int, device: str):
    import torch

    width = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for r, e in enumerate(examples):
        n = len(e["input_ids"])
        ids[r, :n] = torch.tensor(e["input_ids"])
        labels[r, :n] = torch.tensor(e["labels"])
        mask[r, :n] = 1
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device), "labels": labels.to(device)}


def lr_factor(step: int, total: int, warmup: int, floor: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def encode_all(tok, rows: list[dict], max_len: int) -> tuple[list[dict], int]:
    eot = end_of_turn(tok)
    out = [e for e in (encode_example(tok, r, max_len, eot) for r in rows) if e is not None]
    return out, len(rows) - len(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--base-model", default=None, help="override student.base_model")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (for smoke tests)")
    ap.add_argument("--cpu", action="store_true", help="run without a GPU (smoke tests only)")
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoTokenizer

    cfg = load(a.config)
    t = cfg["train"]
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    load_4bit = bool(t["load_4bit"]) and device == "cuda"
    base = a.base_model or cfg["student"]["base_model"]
    out = cfg["root"] / t["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    random.seed(t["seed"])
    torch.manual_seed(t["seed"])

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    train, skipped = encode_all(tok, read_jsonl(cfg["data_dir"] / "paper_train.jsonl"), t["max_seq_len"])
    val, _ = encode_all(tok, read_jsonl(cfg["data_dir"] / "paper_val.jsonl"), t["max_seq_len"])
    if not train:
        raise SystemExit("No training data. Run `python -m paper_distill.build all` first.")
    lengths = [len(e["input_ids"]) for e in train]
    print(f"train: {len(train)} examples ({skipped} too long for max_seq_len and left out), {len(val)} validation, "
          f"{sum(lengths) / 1e6:.2f}M tokens per epoch, device {device}, 4-bit {load_4bit}", flush=True)

    model = load_base(base, load_4bit, device)
    state_path, last = out / "state.pt", out / "last"
    if a.resume and state_path.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(prepare_for_training(model, load_4bit), last, is_trainable=True)
    else:
        model = add_lora(model, t, load_4bit)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"train: {sum(p.numel() for p in params) / 1e6:.1f}M trainable parameters", flush=True)

    per_epoch = [math.ceil(len(make_batches(lengths, t["tokens_per_batch"], random.Random(t["seed"] + e)))
                           / t["grad_accum"]) for e in range(t["epochs"])]
    total = sum(per_epoch)
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, total, t["warmup_steps"], t["min_lr_ratio"]))
    step, start_epoch, start_batch, best = 0, 0, 0, float("inf")
    if a.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu")
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, start_epoch, start_batch, best = st["step"], st["epoch"], st["batch"], st["best"]
        print(f"train: resumed at step {step} (epoch {start_epoch + 1})", flush=True)

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        loss_sum = tok_sum = 0
        vl = [len(e["input_ids"]) for e in val]
        for b in make_batches(vl, t["tokens_per_batch"], random.Random(0)):
            batch = collate([val[i] for i in b], tok.pad_token_id, device)
            with autocast():
                loss = model(**batch).loss
            n = int((batch["labels"] != -100).sum())
            loss_sum, tok_sum = loss_sum + loss.item() * n, tok_sum + n
        model.train()
        return loss_sum / max(1, tok_sum)

    def checkpoint(epoch: int, batch_i: int) -> None:
        nonlocal best
        v = validate() if val else float("nan")
        improved = val and v < best
        if improved or not val:
            best = v if val else best
            model.save_pretrained(out)  # the adapter in the output folder is always the best so far
            tok.save_pretrained(out)
        model.save_pretrained(last)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "epoch": epoch,
                    "batch": batch_i, "best": best}, state_path)
        (out / "paper_meta.json").write_text(json.dumps({
            "base_model": base, "system_prompt": cfg["system_prompt"], "step": step, "best_val_loss": best,
            "train_examples": len(train), "config": {k: v for k, v in t.items()}}, indent=2, default=str))
        print(f"train: step {step}/{total} validation loss {v:.4f}" + (" (best so far, saved)" if improved else ""),
              flush=True)

    model.train()
    t0, seen_tokens, running, done, step0 = time.time(), 0, [], False, step
    stopped_at = (t["epochs"], 0)
    try:
        for epoch in range(start_epoch, t["epochs"]):
            batches = make_batches(lengths, t["tokens_per_batch"], random.Random(t["seed"] + epoch))
            first = start_batch if epoch == start_epoch else 0
            for i in range(first, len(batches)):
                batch = collate([train[j] for j in batches[i]], tok.pad_token_id, device)
                with autocast():
                    loss = model(**batch).loss
                (loss / t["grad_accum"]).backward()
                running.append(loss.item())
                seen_tokens += int(batch["attention_mask"].sum())
                if (i + 1) % t["grad_accum"] == 0 or i + 1 == len(batches):
                    torch.nn.utils.clip_grad_norm_(params, t["max_grad_norm"])
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % t.get("log_every", 10) == 0 or step == 1:
                        el = time.time() - t0
                        rate = seen_tokens / el
                        left = (total - step) * el / max(1, step - step0)
                        print(f"train: step {step}/{total} loss {sum(running) / len(running):.4f} "
                              f"lr {sched.get_last_lr()[0]:.2e} {rate:.0f} tok/s, about {left / 3600:.1f} h left",
                              flush=True)
                        running = []
                    if step % t["eval_every"] == 0:
                        checkpoint(epoch, i + 1)
                    if a.max_steps and step >= a.max_steps:
                        done, stopped_at = True, (epoch, i + 1)
                        break
            if done:
                break
    except torch.cuda.OutOfMemoryError:
        raise SystemExit("Out of GPU memory. Lower train.tokens_per_batch (try 2048) or train.max_seq_len in "
                         "config.yaml, close other programs using the GPU, then run again with --resume.")
    checkpoint(*stopped_at)  # an early stop records where it got to, so --resume carries on from there
    print(f"train: {'stopped early' if done else 'finished'}. Best adapter is in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
