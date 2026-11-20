"""Apple Silicon backend built on MLX / mlx-lm.

Training uses mlx-lm's LoRA layers with a hand-written loop so it shares the engine's
schedule, evaluation, forgetting guard and checkpoint logic. Adapters are written in
mlx-lm's native layout (adapters.safetensors + adapter_config.json, loadable with
`mlx_lm.load(path, adapter_path=...)`) and also converted to PEFT format for export.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator, Optional

from engine.backends.base import Backend


class MlxBackend(Backend):
    name = "mlx"

    def __init__(self, plan: dict):
        super().__init__(plan)
        import mlx.core as mx  # noqa: WPS433
        import mlx.nn as nn  # noqa: WPS433
        import mlx.optimizers as optim  # noqa: WPS433

        self.mx, self.nn, self.optim = mx, nn, optim
        self.model = None
        self.tok = None
        self.optimizer = None
        self._adapter = False
        self._acc = None
        self._loss_and_grad = None
        self.base_model = None  # lazily loaded copy without adapters (for side-by-side)

    # ------------------------------------------------------------------ loading -----------
    def _layers(self) -> Any:
        m = self.model
        for attr in ("layers", "model.layers", "language_model.model.layers"):
            obj = m
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("could not locate transformer layers on this MLX model")

    def load(self, adapter_dir: Optional[Path] = None, trainable: bool = False) -> None:
        from mlx_lm import load as mlx_load  # noqa: WPS433

        path = self.plan["base_model"]["path"]
        if adapter_dir is not None:
            mlx_dir = adapter_dir / "mlx" if (adapter_dir / "mlx").is_dir() else adapter_dir
            self.model, self.tok = mlx_load(path, adapter_path=str(mlx_dir))
            self._adapter = True
            if trainable:
                self.model.freeze()
                self.model.unfreeze(keys=["lora_a", "lora_b", "m"], strict=False)
        else:
            self.model, self.tok = mlx_load(path)
            self._adapter = False

    @property
    def has_adapter(self) -> bool:
        return self._adapter

    def unload(self) -> None:
        self.model = None
        self.base_model = None

    def memory_gb(self) -> Optional[float]:
        try:
            return float(self.mx.get_peak_memory()) / 1024**3
        except AttributeError:
            return float(self.mx.metal.get_peak_memory()) / 1024**3

    # ------------------------------------------------------------------ text --------------
    def render(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, **self.chat_kwargs)

    def _sampler(self, temperature: float, top_p: float) -> Any:
        from mlx_lm.sample_utils import make_sampler  # noqa: WPS433

        return make_sampler(temp=temperature, top_p=top_p)

    def _base(self) -> Any:
        if self.base_model is None:
            from mlx_lm import load as mlx_load  # noqa: WPS433

            self.base_model, _ = mlx_load(self.plan["base_model"]["path"])
        return self.base_model

    def generate(self, convs: list[list[dict]], max_new_tokens: int, temperature: float = 0.0, top_p: float = 1.0, n: int = 1, seed: int = 0, use_adapter: bool = True) -> list[list[str]]:
        from mlx_lm import generate as mlx_generate  # noqa: WPS433

        model = self.model if (use_adapter or not self._adapter) else self._base()
        model.eval()
        sampler = self._sampler(temperature, top_p)
        out = []
        for ci, conv in enumerate(convs):
            prompt = self.render(conv, add_generation_prompt=True)
            outs = []
            for k in range(n):
                self.mx.random.seed(seed + ci * 131 + k)
                outs.append(mlx_generate(model, self.tok, prompt=prompt, max_tokens=max_new_tokens, sampler=sampler, verbose=False).strip())
            out.append(outs)
        return out

    def stream(self, messages: list[dict], max_new_tokens: int, temperature: float = 0.0, top_p: float = 1.0, use_adapter: bool = True) -> Iterator[str]:
        from mlx_lm import stream_generate  # noqa: WPS433

        model = self.model if (use_adapter or not self._adapter) else self._base()
        prompt = self.render(messages, add_generation_prompt=True)
        for resp in stream_generate(model, self.tok, prompt=prompt, max_tokens=max_new_tokens, sampler=self._sampler(temperature, top_p)):
            if resp.text:
                yield resp.text

    # ------------------------------------------------------------------ training ----------
    def _lora_config(self) -> dict:
        lora = self.plan["lora"]
        return {"rank": lora["r"], "scale": lora["alpha"] / lora["r"], "dropout": lora["dropout"], "keys": None}

    def prepare_training(self) -> None:
        from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: WPS433

        if not self._adapter:
            self.model.freeze()
            cfg = self._lora_config()
            layers = self._layers()
            keys = set()
            for name, _ in layers[0].named_modules():
                if name.rsplit(".", 1)[-1] in self.plan["lora"]["targets"]:
                    keys.add(name)
            cfg["keys"] = sorted(keys) or None
            if cfg["keys"] is None:
                cfg.pop("keys")
            linear_to_lora_layers(self.model, len(layers), cfg, use_dora=self.plan["lora"]["dora"])
            # Train ONLY the adapter: some models (e.g. Qwen3's q_norm/k_norm) leave norm weights
            # trainable after conversion, which would then leak into the saved adapter.
            self.model.freeze()
            self.model.unfreeze(keys=["lora_a", "lora_b", "m"], strict=False)
            self._adapter = True
            self._num_layers = len(layers)
            self._keys = cfg.get("keys")
        self.model.train()
        self._loss_and_grad = self.nn.value_and_grad(self.model, self._loss)
        self.reset_optimizer()

    def reset_optimizer(self) -> None:
        self.optimizer = self.optim.AdamW(learning_rate=self.plan["sft"]["lr"], weight_decay=0.0)
        self._acc = None

    def _encode(self, ex: dict) -> Optional[tuple[list[int], int]]:
        msgs = ex["messages"]
        prefix = self.render(msgs[:-1], add_generation_prompt=True)
        full = self.render(msgs, add_generation_prompt=False)
        if not full.startswith(prefix):
            full = prefix + msgs[-1]["content"] + (self.tok.eos_token or "")
        p = self.tok.encode(prefix, add_special_tokens=False)
        f = self.tok.encode(full, add_special_tokens=False)
        max_seq = int(self.plan["sft"]["max_seq"])
        start = len(p)
        if len(f) > max_seq:
            cut = len(f) - max_seq
            f = f[cut:]
            start = max(0, start - cut)
        if start >= len(f) - 1:
            return None
        return f, start

    def _arrays(self, examples: list[dict]) -> Optional[tuple[Any, Any, Any]]:
        enc = [e for e in (self._encode(x) for x in examples) if e is not None]
        if not enc:
            return None
        L = max(len(ids) for ids, _ in enc)
        pad = self.tok.pad_token_id if getattr(self.tok, "pad_token_id", None) is not None else 0
        ids = [x + [pad] * (L - len(x)) for x, _ in enc]
        mask = [[1 if (start <= j < len(x) - 1) else 0 for j in range(L - 1)] for x, start in enc]
        a = self.mx.array(ids)
        return a[:, :-1], a[:, 1:], self.mx.array(mask)

    def _loss(self, model: Any, inputs: Any, targets: Any, mask: Any) -> Any:
        logits = model(inputs).astype(self.mx.float32)
        ce = self.nn.losses.cross_entropy(logits, targets) * mask
        return ce.sum() / self.mx.maximum(mask.sum(), 1)

    def train_microbatch(self, examples: list[dict], scale: float) -> tuple[float, int]:
        from mlx.utils import tree_map  # noqa: WPS433

        arrs = self._arrays(examples)
        if arrs is None:
            return 0.0, 0
        x, y, m = arrs
        loss, grads = self._loss_and_grad(self.model, x, y, m)
        grads = tree_map(lambda g: g * scale, grads)
        self._acc = grads if self._acc is None else tree_map(lambda a, b: a + b, self._acc, grads)
        self.mx.eval(loss, self._acc)
        value = float(loss.item())
        if not math.isfinite(value):
            raise FloatingPointError("loss became NaN/inf")
        return value, int(m.sum().item())

    def optimizer_step(self, lr: float, max_grad_norm: float) -> float:
        if self._acc is None:
            return 0.0
        grads, norm = self.optim.clip_grad_norm(self._acc, max_grad_norm)
        self.optimizer.learning_rate = lr
        self.optimizer.update(self.model, grads)
        self.mx.eval(self.model.parameters(), self.optimizer.state)
        self._acc = None
        return float(norm.item())

    def nll(self, examples: list[dict]) -> float:
        total, count = 0.0, 0
        for i in range(0, len(examples), 8):
            arrs = self._arrays(examples[i : i + 8])
            if arrs is None:
                continue
            x, y, m = arrs
            logits = self.model(x).astype(self.mx.float32)
            ce = (self.nn.losses.cross_entropy(logits, y) * m).sum()
            total += float(ce.item())
            count += int(m.sum().item())
        return total / count if count else math.nan

    def _adapter_weights(self) -> dict:
        from mlx.utils import tree_flatten  # noqa: WPS433

        return {k: v for k, v in tree_flatten(self.model.trainable_parameters()) if k.rsplit(".", 1)[-1] in ("lora_a", "lora_b", "m")}

    def _write_mlx(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        self.mx.save_safetensors(str(d / "adapters.safetensors"), self._adapter_weights())
        lora = self.plan["lora"]
        cfg = {
            "fine_tune_type": "dora" if lora["dora"] else "lora",
            "num_layers": getattr(self, "_num_layers", len(self._layers())),
            "lora_parameters": {"rank": lora["r"], "scale": lora["alpha"] / lora["r"], "dropout": lora["dropout"], **({"keys": self._keys} if getattr(self, "_keys", None) else {})},
        }
        (d / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def save_checkpoint(self, d: Path) -> None:
        from mlx.utils import tree_flatten  # noqa: WPS433

        self._write_mlx(d)
        if self.optimizer is not None:
            state = dict(tree_flatten(self.optimizer.state))
            if state:
                self.mx.save_safetensors(str(d / "optimizer.safetensors"), state)

    def load_checkpoint(self, d: Path) -> None:
        from mlx.utils import tree_unflatten  # noqa: WPS433

        weights = self.mx.load(str(d / "adapters.safetensors"))
        self.model.load_weights(list(weights.items()), strict=False)
        opt = d / "optimizer.safetensors"
        if opt.is_file() and self.optimizer is not None:
            self.optimizer.state = tree_unflatten(list(self.mx.load(str(opt)).items()))

    def save_adapter(self, d: Path) -> None:
        from engine.adapters import mlx_to_peft  # noqa: WPS433

        self._write_mlx(d / "mlx")
        mlx_to_peft(d / "mlx", d, self.plan)
        self.write_adapter_meta(d)
