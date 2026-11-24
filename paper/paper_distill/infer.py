"""Run Paper, with the inference-time strategies that help a small model on multi-step problems.

    python -m paper_distill.infer "your question"                       # adaptive majority vote (default)
    python -m paper_distill.infer "your question" --strategy single     # one sample, fastest
    python -m paper_distill.infer "your question" --http http://localhost:11434/v1 --name paper

Strategies:
  single  one reasoning pass.
  vote    sample a few independent solutions and return the answer most of them reach.
          Sampling stops early once enough of them agree.
Both use a reasoning budget: if the model is still thinking when the budget runs out,
the think block is closed and it has to answer.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .chat import end_of_turn, prompt_text
from .config import load
from .schema import THINK_CLOSE, THINK_OPEN
from .teacher import Teacher, split_reasoning
from .verify import canonical, extract_final_answer


class HFBackend:
    """The student loaded with transformers: a merged model folder, or a base model plus LoRA adapter."""

    def __init__(self, model: str, adapter: str | None = None, load_4bit: bool = False, cpu: bool = False):
        import torch
        from transformers import AutoTokenizer

        from .train import load_base
        self.torch = torch
        self.device = "cpu" if cpu or not torch.cuda.is_available() else "cuda"
        self.tok = AutoTokenizer.from_pretrained(adapter or model)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = load_base(model, load_4bit and self.device == "cuda", self.device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.eot = end_of_turn(self.tok)

    def _generate(self, texts: list[str], max_new: int, temperature: float, top_p: float, stop=None):
        """Continue each text. Returns (new_text, finished) pairs; finished means the turn was ended."""
        torch = self.torch
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        kw = dict(max_new_tokens=max_new, pad_token_id=self.tok.pad_token_id, do_sample=temperature > 0)
        if temperature > 0:
            kw.update(temperature=temperature, top_p=top_p)
        if stop:
            kw.update(stop_strings=stop, tokenizer=self.tok)
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        results = []
        for row in out[:, enc["input_ids"].shape[1]:]:
            text = self.tok.decode(row, skip_special_tokens=False)
            finished = self.eot in text
            text = text.split(self.eot, 1)[0]
            for special in {self.tok.pad_token, self.tok.eos_token} - {None}:
                text = text.replace(special, "")
            results.append((text, finished))
        return results

    def complete(self, prompts: list[str], *, temperature: float, top_p: float, max_new_tokens: int,
                 think_budget: int) -> list[str]:
        """Full assistant text (think block and answer) for each prompt, under a reasoning budget."""
        first = self._generate(prompts, min(think_budget, max_new_tokens), temperature, top_p, stop=[THINK_CLOSE])
        done: dict[int, str] = {}
        pending: list[tuple[int, str]] = []
        for i, (text, finished) in enumerate(first):
            if finished:
                done[i] = text
            elif THINK_CLOSE in text:
                pending.append((i, text))
            else:  # budget spent while still reasoning: close the block and make it answer
                pending.append((i, text.rstrip() + f"\n{THINK_CLOSE}\n\n"))
        if pending:
            rest = max(64, max_new_tokens - think_budget)
            second = self._generate([prompts[i] + t for i, t in pending], rest, temperature, top_p)
            for (i, t), (more, _) in zip(pending, second):
                done[i] = t + more
        return [done[i] for i in range(len(prompts))]

    def ask(self, system: str, questions: list[str], n: int = 1, **gen) -> list[list[str]]:
        prompts = [prompt_text(self.tok, system, q) for q in questions for _ in range(n)]
        flat = self.complete(prompts, **gen)
        return [flat[i * n:(i + 1) * n] for i in range(len(questions))]


class HTTPBackend:
    """Any OpenAI-compatible server: Paper exported to Ollama or llama.cpp, or the teacher itself."""

    def __init__(self, cfg: dict, system_in_user: bool = False):
        self.teacher, self.system_in_user = Teacher(cfg), system_in_user

    def ask(self, system: str, questions: list[str], n: int = 1, *, temperature: float, top_p: float,
            max_new_tokens: int, think_budget: int) -> list[list[str]]:
        out = []
        for q in questions:
            msgs = ([{"role": "user", "content": f"{system}\n\n{q}"}] if self.system_in_user else
                    [{"role": "system", "content": system}, {"role": "user", "content": q}])
            row = []
            for _ in range(n):
                r = self.teacher.chat(msgs, max_tokens=max_new_tokens, temperature=temperature)
                row.append(f"{THINK_OPEN}\n{r.reasoning}\n{THINK_CLOSE}\n\n{r.content}")
            out.append(row)
        return out


def parse(text: str) -> tuple[str, str, str | None]:
    """(reasoning, visible answer, extracted final answer) from one assistant text."""
    reasoning, answer = split_reasoning({"content": text})
    if THINK_CLOSE in answer:  # the model reopened its reasoning; the answer is what follows the last close
        extra, _, answer = answer.rpartition(THINK_CLOSE)
        reasoning = f"{reasoning}\n{extra.replace(THINK_OPEN, '')}".strip()
        answer = answer.strip()
    return reasoning, answer, extract_final_answer(answer, strict=False)


def tally(texts: list[str]) -> tuple[Counter, dict[str, str]]:
    """Votes per distinct answer, and the first sample that gave each answer."""
    votes, first = Counter(), {}
    for t in texts:
        key = canonical(parse(t)[2])
        if key is not None:
            votes[key] += 1
            first.setdefault(key, t)
    return votes, first


def solve(backend, system: str, question: str, inf: dict, strategy: str = "vote") -> dict:
    gen = dict(temperature=inf["temperature"], top_p=inf["top_p"], max_new_tokens=inf["max_new_tokens"],
               think_budget=inf["think_budget"])
    if strategy == "single":
        texts = backend.ask(system, [question], 1, **gen)[0]
    else:
        k = min(inf["early_stop"], inf["votes"])
        texts = backend.ask(system, [question], k, **gen)[0]
        votes, _ = tally(texts)
        if not votes or votes.most_common(1)[0][1] < k:  # no clear agreement yet: draw the rest
            if inf["votes"] > k:
                texts += backend.ask(system, [question], inf["votes"] - k, **gen)[0]
    votes, first = tally(texts)
    if votes:
        top = max(votes.values())
        winner = next(k for k in first if votes[k] == top)  # ties go to the answer seen first
        chosen = first[winner]
    else:
        chosen = texts[0]
    reasoning, answer, final = parse(chosen)
    return {"final_answer": final, "answer": answer, "reasoning": reasoning, "samples": len(texts),
            "agreement": (max(votes.values()) / len(texts)) if votes else 0.0,
            "votes": {parse(first[k])[2]: n for k, n in votes.most_common()}}


def backend_from_args(cfg: dict, a) -> tuple[object, str]:
    """Pick the model to run from command-line arguments. Returns (backend, system prompt)."""
    if a.http:
        tc = {**cfg["teacher"], "base_url": a.http, "model": a.name, "extra_body": {}, "direct_body": {},
              "temperature": cfg["inference"]["temperature"]}
        return HTTPBackend(tc), cfg["system_prompt"]
    out = cfg["root"] / cfg["train"]["output_dir"]
    merged = cfg["root"] / "out" / "paper-merged"
    if a.model:
        return HFBackend(a.model, a.adapter, cfg["train"]["load_4bit"], a.cpu), cfg["system_prompt"]
    if merged.exists():
        return HFBackend(str(merged), None, cfg["train"]["load_4bit"], a.cpu), cfg["system_prompt"]
    if (out / "adapter_config.json").exists():
        return HFBackend(cfg["student"]["base_model"], str(out), cfg["train"]["load_4bit"], a.cpu), cfg["system_prompt"]
    raise SystemExit("No trained model found. Run `python -m paper_distill.train` first, or pass --model.")


def add_model_args(ap) -> None:
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None, help="a model folder or hub id (default: the trained Paper)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter folder to apply to --model")
    ap.add_argument("--http", default=None, help="OpenAI-compatible base URL to use instead of loading a model")
    ap.add_argument("--name", default="paper", help="model name on the --http server")
    ap.add_argument("--cpu", action="store_true")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--strategy", choices=["single", "vote"], default="vote")
    ap.add_argument("--show-reasoning", action="store_true")
    ap.add_argument("--json", action="store_true")
    add_model_args(ap)
    a = ap.parse_args(argv)
    cfg = load(a.config)
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    backend, system = backend_from_args(cfg, a)
    res = solve(backend, system, a.question, cfg["inference"], a.strategy)
    if a.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    if a.show_reasoning:
        print("--- reasoning ---\n" + res["reasoning"] + "\n")
    print(res["answer"])
    if a.strategy == "vote":
        print(f"\n[{res['samples']} samples, {res['agreement']:.0%} agreed; votes: {res['votes']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
