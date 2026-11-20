"""Inference-only backend over llama.cpp (llama-cpp-python): GGUF base + optional GGUF LoRA."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from engine.backends.base import Backend


class LlamaCppBackend(Backend):
    name = "llamacpp"
    can_train = False

    def __init__(self, plan: dict):
        super().__init__(plan)
        self.llm = None
        self.base_llm = None
        self._adapter = False

    def _open(self, lora: Optional[str]) -> object:
        from llama_cpp import Llama  # noqa: WPS433

        return Llama(model_path=self.plan["base_model"]["gguf"], lora_path=lora, n_ctx=int(self.plan["sft"].get("max_seq", 4096)), n_gpu_layers=-1, verbose=False)

    def load(self, adapter_dir: Optional[Path] = None, trainable: bool = False) -> None:
        lora = None
        if adapter_dir is not None:
            ggufs = sorted(Path(adapter_dir).glob("*.gguf"))
            lora = str(ggufs[0]) if ggufs else None
        self.llm = self._open(lora)
        self._adapter = lora is not None

    @property
    def has_adapter(self) -> bool:
        return self._adapter

    def render(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def _llm(self, use_adapter: bool) -> object:
        if use_adapter or not self._adapter:
            return self.llm
        if self.base_llm is None:
            self.base_llm = self._open(None)
        return self.base_llm

    def generate(self, convs: list[list[dict]], max_new_tokens: int, temperature: float = 0.0, top_p: float = 1.0, n: int = 1, seed: int = 0, use_adapter: bool = True) -> list[list[str]]:
        llm = self._llm(use_adapter)
        out = []
        for conv in convs:
            outs = []
            for k in range(n):
                r = llm.create_chat_completion(messages=conv, max_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=seed + k)  # type: ignore[attr-defined]
                outs.append(r["choices"][0]["message"]["content"].strip())
            out.append(outs)
        return out

    def stream(self, messages: list[dict], max_new_tokens: int, temperature: float = 0.0, top_p: float = 1.0, use_adapter: bool = True) -> Iterator[str]:
        llm = self._llm(use_adapter)
        for chunk in llm.create_chat_completion(messages=messages, max_tokens=max_new_tokens, temperature=temperature, top_p=top_p, stream=True):  # type: ignore[attr-defined]
            delta = chunk["choices"][0].get("delta", {}).get("content")
            if delta:
                yield delta
