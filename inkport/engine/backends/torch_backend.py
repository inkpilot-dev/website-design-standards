"""PyTorch backend: transformers + peft (LoRA/DoRA), optional bitsandbytes QLoRA and Unsloth.

Devices: CUDA (NVIDIA and AMD ROCm builds), Intel XPU, Apple MPS, CPU.
Adapters are saved in the canonical PEFT format (adapter_model.safetensors + adapter_config.json).
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

from engine import events
from engine.backends.base import Backend


def _import_unsloth() -> Any:
    # Unsloth must be imported before transformers so its patches apply.
    import unsloth  # type: ignore  # noqa: F401,WPS433
    from unsloth import FastLanguageModel  # type: ignore  # noqa: WPS433

    return FastLanguageModel


def load_tokenizer(plan: dict) -> Any:
    """Tokenizer files shipped next to the model win; otherwise read it from the GGUF."""
    from transformers import AutoTokenizer  # noqa: WPS433

    path = plan["base_model"]["path"]
    gguf = plan["base_model"].get("gguf")
    if gguf and not (Path(path) / "tokenizer.json").is_file():
        return AutoTokenizer.from_pretrained(path, gguf_file=gguf)
    return AutoTokenizer.from_pretrained(path)


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, plan: dict, use_unsloth: bool = False):
        super().__init__(plan)
        self.use_unsloth = use_unsloth
        self.FastLanguageModel = None
        if use_unsloth:
            try:
                self.FastLanguageModel = _import_unsloth()
                self.name = "unsloth"
            except Exception as exc:  # noqa: BLE001
                events.log(f"Unsloth unavailable ({exc}); using transformers + peft", "warn")
                self.use_unsloth = False
        import torch  # noqa: WPS433

        self.torch = torch
        self.device = plan.get("device") or self._pick_device()
        self.dtype = self._pick_dtype()
        self.model = None
        self.tok = None
        self.optimizer = None
        self._gen_lock = threading.Lock()

    # ------------------------------------------------------------------ setup -------------
    def _pick_device(self) -> str:
        t = self.torch
        if t.cuda.is_available():
            return "cuda"
        if hasattr(t, "xpu") and t.xpu.is_available():
            return "xpu"
        if hasattr(t.backends, "mps") and t.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _pick_dtype(self) -> Any:
        t = self.torch
        q = self.plan.get("quant", "bf16")
        if self.device == "cpu" or q == "fp32":
            return t.float32
        if self.device == "cuda":
            return t.bfloat16 if t.cuda.is_bf16_supported() else t.float16
        if self.device == "mps":
            return t.float16
        return t.bfloat16

    def load(self, adapter_dir: Optional[Path] = None, trainable: bool = False) -> None:
        path = self.plan["base_model"]["path"]
        quant = self.plan.get("quant", "bf16")
        if self.use_unsloth and self.FastLanguageModel is not None:
            self.model, self.tok = self.FastLanguageModel.from_pretrained(
                model_name=str(adapter_dir) if adapter_dir else path,
                max_seq_length=int(self.plan["sft"]["max_seq"]),
                load_in_4bit=quant == "nf4",
                dtype=None,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433

            self.tok = load_tokenizer(self.plan)
            kwargs: dict[str, Any] = {"dtype": self.dtype, "low_cpu_mem_usage": True}
            if quant == "nf4" and self.device == "cuda":
                from transformers import BitsAndBytesConfig  # noqa: WPS433

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=self.dtype,
                )
                kwargs["device_map"] = {"": 0}
            gguf = self.plan["base_model"].get("gguf")
            if gguf:
                # Bundled 4-bit GGUF starter: dequantized in memory to a trainable dtype; the
                # same file is what llama.cpp serves, so the adapter targets the shipped weights.
                kwargs.pop("quantization_config", None)
                kwargs.pop("device_map", None)
                self.model = AutoModelForCausalLM.from_pretrained(path, gguf_file=gguf, **kwargs)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
            if "device_map" not in kwargs:
                self.model.to(self.device)
            if adapter_dir is not None:
                from peft import PeftModel  # noqa: WPS433

                self.model = PeftModel.from_pretrained(self.model, str(adapter_dir), is_trainable=trainable)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model.eval()

    @property
    def has_adapter(self) -> bool:
        return hasattr(self.model, "peft_config")

    def unload(self) -> None:
        self.model = None
        self.optimizer = None
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    def memory_gb(self) -> Optional[float]:
        if self.device == "cuda":
            return self.torch.cuda.max_memory_allocated() / 1024**3
        return None

    # ------------------------------------------------------------------ text --------------
    def render(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, **self.chat_kwargs)

    def _encode_example(self, ex: dict, max_seq: int) -> Optional[tuple[list[int], list[int]]]:
        msgs = ex["messages"]
        prefix = self.render(msgs[:-1], add_generation_prompt=True)
        if ex.get("open_end"):
            full = prefix + msgs[-1]["content"]  # the answer alone, without what marks its end
        else:
            full = self.render(msgs, add_generation_prompt=False)
            if not full.startswith(prefix):
                full = prefix + msgs[-1]["content"] + (self.tok.eos_token or "")
        p_ids = self.tok(prefix, add_special_tokens=False)["input_ids"]
        f_ids = self.tok(full, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(p_ids) + f_ids[len(p_ids) :]
        if len(f_ids) > max_seq:
            cut = len(f_ids) - max_seq
            f_ids, labels = f_ids[cut:], labels[cut:]
        if all(lab == -100 for lab in labels):
            return None
        return f_ids, labels

    def _batch(self, examples: list[dict]) -> Optional[dict]:
        max_seq = int(self.plan["sft"]["max_seq"])
        enc = [e for e in (self._encode_example(x, max_seq) for x in examples) if e is not None]
        if not enc:
            return None
        L = max(len(i) for i, _ in enc)
        pad = self.tok.pad_token_id
        t = self.torch
        ids = t.tensor([i + [pad] * (L - len(i)) for i, _ in enc], device=self.device)
        lab = t.tensor([lb + [-100] * (L - len(lb)) for _, lb in enc], device=self.device)
        att = t.tensor([[1] * len(i) + [0] * (L - len(i)) for i, _ in enc], device=self.device)
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    def generate(self, convs: list[list[dict]], max_new_tokens: int, temperature: float = 0.0, top_p: float = 1.0, n: int = 1, seed: int = 0, use_adapter: bool = True) -> list[list[str]]:
        t = self.torch
        out: list[list[str]] = []
        # On a graphics card answers are written many at a time: a run is mostly answering (the
        # exams, the checks, the practice rounds), and eight at a time took 25 to 30 minutes for a
        # run on a 16 GB card whose training steps took a small part of that. If the card runs
        # out of memory the batch is halved and tried again.
        bs = 16 if self.device == "cpu" else getattr(self, "_gen_batch", 32)
        was_training = self.model.training
        self.model.eval()
        if self.FastLanguageModel is not None:
            self.FastLanguageModel.for_inference(self.model)
        try:
            with self._gen_lock, t.inference_mode():
                i = 0
                while i < len(convs):
                    chunk = convs[i : i + bs]
                    prompts = [self.render(c, add_generation_prompt=True) for c in chunk]
                    enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
                    t.manual_seed(seed + i)
                    kwargs: dict[str, Any] = {
                        "max_new_tokens": max_new_tokens,
                        "pad_token_id": self.tok.pad_token_id,
                        "eos_token_id": self.tok.eos_token_id,
                        "num_return_sequences": n,
                    }
                    if temperature > 0:
                        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
                    else:
                        kwargs.update(do_sample=False)
                    try:
                        if not use_adapter and self.has_adapter:
                            with self.model.disable_adapter():
                                gen = self.model.generate(**enc, **kwargs)
                        else:
                            gen = self.model.generate(**enc, **kwargs)
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower() or bs <= 1 or self.device == "cpu":
                            raise
                        del enc
                        t.cuda.empty_cache()
                        bs = max(1, bs // 2)
                        self._gen_batch = bs  # remembered for the rest of the run
                        continue
                    new = gen[:, enc["input_ids"].shape[1] :]
                    texts = self.tok.batch_decode(new, skip_special_tokens=True)
                    for j in range(len(chunk)):
                        out.append([s.strip() for s in texts[j * n : (j + 1) * n]])
                    i += len(chunk)
        finally:
            if self.FastLanguageModel is not None and was_training:
                self.FastLanguageModel.for_training(self.model)
            if was_training:
                self.model.train()
        return out

    def stream(self, messages: list[dict], max_new_tokens: int, temperature: float = 0.0, top_p: float = 1.0, use_adapter: bool = True) -> Iterator[str]:
        from transformers import TextIteratorStreamer  # noqa: WPS433

        prompt = self.render(messages, add_generation_prompt=True)
        enc = self.tok([prompt], return_tensors="pt", add_special_tokens=False).to(self.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tok.pad_token_id, "eos_token_id": self.tok.eos_token_id, "streamer": streamer}
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p) if temperature > 0 else kwargs.update(do_sample=False)

        def run() -> None:
            with self._gen_lock, self.torch.inference_mode():
                if not use_adapter and self.has_adapter:
                    with self.model.disable_adapter():
                        self.model.generate(**enc, **kwargs)
                else:
                    self.model.generate(**enc, **kwargs)

        th = threading.Thread(target=run, daemon=True)
        th.start()
        for piece in streamer:
            if piece:
                yield piece
        th.join()

    # ------------------------------------------------------------------ training ----------
    def _target_modules(self) -> Any:
        wanted = set(self.plan["lora"]["targets"])
        present = {name.rsplit(".", 1)[-1] for name, mod in self.model.named_modules() if mod.__class__.__name__ in ("Linear", "Linear4bit", "Linear8bitLt")}
        hit = sorted(wanted & present)
        return hit if hit else "all-linear"

    def prepare_training(self) -> None:
        lora = self.plan["lora"]
        sft = self.plan["sft"]
        if not self.has_adapter:
            if self.FastLanguageModel is not None:
                self.model = self.FastLanguageModel.get_peft_model(
                    self.model,
                    r=lora["r"],
                    lora_alpha=lora["alpha"],
                    lora_dropout=lora["dropout"],
                    target_modules=self._target_modules(),
                    use_gradient_checkpointing="unsloth" if sft["gradient_checkpointing"] else False,
                    random_state=self.plan["seed"],
                    use_dora=lora["dora"],
                )
            else:
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: WPS433

                if self.plan.get("quant") == "nf4" and self.device == "cuda":
                    self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=sft["gradient_checkpointing"])
                cfg = LoraConfig(
                    r=lora["r"],
                    lora_alpha=lora["alpha"],
                    lora_dropout=lora["dropout"],
                    target_modules=self._target_modules(),
                    use_dora=lora["dora"],
                    task_type="CAUSAL_LM",
                )
                self.torch.manual_seed(self.plan["seed"])
                self.model = get_peft_model(self.model, cfg)
        if sft["gradient_checkpointing"] and self.FastLanguageModel is None and self.device != "cpu":
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        self.model.train()
        self.reset_optimizer()

    def reset_optimizer(self) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt_cls = self.torch.optim.AdamW
        if self.device == "cuda":
            try:
                import bitsandbytes as bnb  # noqa: WPS433

                opt_cls = bnb.optim.PagedAdamW8bit
            except Exception:  # noqa: BLE001
                pass
        self.optimizer = opt_cls(params, lr=self.plan["sft"]["lr"], weight_decay=0.0)
        self.optimizer.zero_grad(set_to_none=True)

    def train_microbatch(self, examples: list[dict], scale: float) -> tuple[float, int]:
        batch = self._batch(examples)
        if batch is None:
            return 0.0, 0
        out = self.model(**batch)
        loss = out.loss
        if not self.torch.isfinite(loss):
            raise FloatingPointError("loss became NaN/inf")
        (loss * scale).backward()
        ntok = int((batch["labels"] != -100).sum().item())
        return float(loss.detach().item()), ntok

    def optimizer_step(self, lr: float, max_grad_norm: float) -> float:
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        params = [p for p in self.model.parameters() if p.requires_grad]
        norm = self.torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(norm)

    def nll(self, examples: list[dict]) -> float:
        total, count = 0.0, 0
        was_training = self.model.training
        self.model.eval()
        with self.torch.no_grad():
            for i in range(0, len(examples), 8):
                batch = self._batch(examples[i : i + 8])
                if batch is None:
                    continue
                logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1].float()
                labels = batch["labels"][:, 1:]
                mask = labels != -100
                lp = self.torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.clamp(min=0).reshape(-1), reduction="none").reshape(labels.shape)
                total += float((lp * mask).sum().item())
                count += int(mask.sum().item())
        if was_training:
            self.model.train()
        return total / count if count else math.nan

    def save_checkpoint(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(d))
        if self.optimizer is not None:
            self.torch.save(self.optimizer.state_dict(), d / "optimizer.pt")

    def load_checkpoint(self, d: Path) -> None:
        from peft import set_peft_model_state_dict  # noqa: WPS433
        from safetensors.torch import load_file  # noqa: WPS433

        set_peft_model_state_dict(self.model, load_file(str(d / "adapter_model.safetensors"), device=str(self.device) if self.device != "mps" else "cpu"))
        opt = d / "optimizer.pt"
        if opt.is_file() and self.optimizer is not None:
            self.optimizer.load_state_dict(self.torch.load(opt, map_location=self.device, weights_only=True))

    def save_adapter(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(d))
        self.write_adapter_meta(d)

    # ------------------------------------------------------------------ GRPO (TRL) --------
    def grpo(self, prompts: list[list[dict]], reward_fn: Any, cfg: dict, out_dir: Path, progress: Any) -> dict:
        """Online RL with TRL's GRPOTrainer. reward_fn(indices, completions) -> list[float]."""
        from datasets import Dataset  # noqa: WPS433
        from transformers import TrainerCallback  # noqa: WPS433
        from trl import GRPOConfig, GRPOTrainer  # noqa: WPS433

        texts = [self.render(p, add_generation_prompt=True) for p in prompts]
        ds = Dataset.from_dict({"prompt": texts, "item_index": list(range(len(texts)))})

        def reward(prompts: list, completions: list, item_index: list, **_: Any) -> list[float]:
            comps = [c if isinstance(c, str) else c[-1]["content"] for c in completions]
            return reward_fn(list(item_index), comps)

        total_steps = int(cfg["max_steps"])

        class _Cb(TrainerCallback):
            def on_log(self, args: Any, state: Any, control: Any, logs: Optional[dict] = None, **kw: Any) -> None:
                logs = logs or {}
                progress.update(state.global_step, {"reward": logs.get("reward"), "loss": logs.get("loss"), "kl": logs.get("kl")})

        g = cfg["num_generations"]
        args = GRPOConfig(
            output_dir=str(out_dir),
            learning_rate=cfg["lr"],
            per_device_train_batch_size=g,
            gradient_accumulation_steps=cfg.get("grad_accum", 1),
            num_generations=g,
            max_completion_length=cfg["max_completion_length"],
            beta=cfg.get("beta", 0.02),
            loss_type=cfg.get("loss_type", "dr_grpo"),
            scale_rewards=False,
            epsilon=0.2,
            epsilon_high=0.28,
            temperature=cfg.get("temperature", 1.0),
            max_steps=total_steps,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            seed=self.plan["seed"],
            bf16=self.dtype == self.torch.bfloat16 and self.device == "cuda",
            use_cpu=self.device == "cpu",
            gradient_checkpointing=bool(self.plan["sft"]["gradient_checkpointing"]) and self.device != "cpu",
        )
        trainer = GRPOTrainer(model=self.model, reward_funcs=[reward], args=args, train_dataset=ds, processing_class=self.tok, callbacks=[_Cb()])
        result = trainer.train()
        self.model = trainer.model
        self.model.eval()
        return {"steps": int(result.global_step), "loss": float(result.training_loss)}


def write_merged(plan: dict, adapter_dir: Path, out_dir: Path) -> Path:
    """Merge the adapter into a full-precision copy of the base and save safetensors."""
    import torch  # noqa: WPS433
    from peft import PeftModel  # noqa: WPS433
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433

    extra = {"gguf_file": plan["base_model"]["gguf"]} if plan["base_model"].get("gguf") else {}
    base = AutoModelForCausalLM.from_pretrained(plan["base_model"]["path"], dtype=torch.float32 if plan.get("quant") == "fp32" else torch.bfloat16, low_cpu_mem_usage=True, **extra)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    load_tokenizer(plan).save_pretrained(str(out_dir))
    (out_dir / "inkport_merged.json").write_text(json.dumps({"base": plan["base_model"]["id"], "adapter": str(adapter_dir)}), encoding="utf-8")
    return out_dir
