"""Backend interface shared by the torch (transformers/peft/unsloth), MLX and llama.cpp backends.

The training *loop* (schedule, accumulation, evaluation cadence, forgetting guard, early
stopping, checkpoints) lives in engine/pipeline/train_and_gate.py and is backend-agnostic;
backends only implement tensor-level primitives.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional


class Backend(ABC):
    name = "base"
    can_train = True

    def __init__(self, plan: dict):
        self.plan = plan
        self.chat_kwargs = dict(plan.get("base_model", {}).get("chat_kwargs") or {})

    # -- loading -------------------------------------------------------------------------
    @abstractmethod
    def load(self, adapter_dir: Optional[Path] = None, trainable: bool = False) -> None: ...

    @property
    @abstractmethod
    def has_adapter(self) -> bool: ...

    def unload(self) -> None:
        pass

    # -- text ----------------------------------------------------------------------------
    @abstractmethod
    def render(self, messages: list[dict], add_generation_prompt: bool = True) -> str: ...

    @abstractmethod
    def generate(self, convs: list[list[dict]], max_new_tokens: int, temperature: float = 0.0, top_p: float = 1.0, n: int = 1, seed: int = 0, use_adapter: bool = True) -> list[list[str]]:
        """Returns n completions per conversation."""

    def stream(self, messages: list[dict], max_new_tokens: int, temperature: float = 0.0, top_p: float = 1.0, use_adapter: bool = True) -> Iterator[str]:
        yield self.generate([messages], max_new_tokens, temperature, top_p, 1, 0, use_adapter)[0][0]

    # -- training ------------------------------------------------------------------------
    def prepare_training(self) -> None:
        raise NotImplementedError

    def reset_optimizer(self) -> None:
        raise NotImplementedError

    def train_microbatch(self, examples: list[dict], scale: float) -> tuple[float, int]:
        raise NotImplementedError

    def optimizer_step(self, lr: float, max_grad_norm: float) -> float:
        raise NotImplementedError

    def nll(self, examples: list[dict]) -> float:
        raise NotImplementedError

    def save_checkpoint(self, d: Path) -> None:
        raise NotImplementedError

    def load_checkpoint(self, d: Path) -> None:
        raise NotImplementedError

    def save_adapter(self, d: Path) -> None:
        raise NotImplementedError

    def memory_gb(self) -> Optional[float]:
        return None

    # -- helpers -------------------------------------------------------------------------
    def write_adapter_meta(self, d: Path) -> None:
        meta = {
            "backend": self.name,
            "base_model_id": self.plan["base_model"]["id"],
            "base_model_path": self.plan["base_model"]["path"],
            "quant": self.plan.get("quant"),
            "lora": self.plan.get("lora"),
        }
        (d / "inkport_adapter.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def make_backend(plan: dict, inference_only: bool = False) -> Backend:
    name = plan["backend"]
    if name == "mlx":
        from engine.backends.mlx_backend import MlxBackend  # noqa: WPS433

        return MlxBackend(plan)
    if name == "llamacpp":
        from engine.backends.llamacpp_backend import LlamaCppBackend  # noqa: WPS433

        return LlamaCppBackend(plan)
    from engine.backends.torch_backend import TorchBackend  # noqa: WPS433

    return TorchBackend(plan, use_unsloth=(name == "unsloth" and not inference_only))
