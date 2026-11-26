"""Train, resume, merge and generate on a tiny randomly initialised model, on the CPU.

This exercises the real code path (chat template, loss masking, LoRA, checkpoints,
budgeted generation) in about a minute. It needs torch, transformers and peft, and
the student's tokenizer: set PAPER_TOKENIZER to a local folder, or let it download.
It proves the plumbing works. It says nothing about the quality of a real run.
"""
import json
import os
import re
import shutil
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from paper_distill import build, evaluate, export, infer, train  # noqa: E402
from paper_distill.config import load, read_jsonl  # noqa: E402
from test_build import ScriptedTeacher  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get("PAPER_TOKENIZER", "Qwen/Qwen3-4B")


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    try:
        tok = transformers.AutoTokenizer.from_pretrained(SOURCE)
        mc = transformers.AutoConfig.from_pretrained(SOURCE)
    except Exception as e:  # no network and no local copy
        pytest.skip(f"student tokenizer not available: {e}")
    root = tmp_path_factory.mktemp("paper")
    tiny = root / "tiny"
    # Same architecture and vocabulary as the real student, shrunk to two small layers.
    mc = type(mc)(vocab_size=mc.vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, intermediate_size=128, head_dim=16, max_position_embeddings=2048,
                  bos_token_id=mc.bos_token_id, eos_token_id=mc.eos_token_id, tie_word_embeddings=True)
    torch.manual_seed(0)
    transformers.AutoModelForCausalLM.from_config(mc).save_pretrained(tiny)
    tok.save_pretrained(tiny)

    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw["dataset"].update(target_size=120, min_size=100, eval_holdout=24, oversample=2.5,
                          mix={"procedural": 1.0, "teacher_authored": 0.0, "public": 0.0})
    raw["student"]["base_model"] = str(tiny)
    raw["train"].update(load_4bit=False, max_seq_len=512, tokens_per_batch=1536, grad_accum=2, epochs=8, lr=1e-2,
                        warmup_steps=2, eval_every=5, log_every=1, lora_r=8, lora_alpha=16)
    raw["inference"].update(max_new_tokens=40, think_budget=16, votes=3, early_stop=2)
    (root / "config.yaml").write_text(yaml.safe_dump(raw))
    cfg = load(root / "config.yaml")
    build.prepare(cfg)
    sheet = {s["question"]: s["ground_truth"] for s in read_jsonl(build.paths(cfg)["procedural"])}
    build.solve(cfg, ScriptedTeacher(sheet))
    build.select(cfg)
    assert build.audit(cfg)
    return root


def losses(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"train: step \d+/\d+ loss ([\d.]+)", text)]


def test_train_resume_merge_generate(project, capsys):
    cfgp = str(project / "config.yaml")
    out = project / "out" / "paper-lora"

    assert train.main(["--config", cfgp, "--cpu", "--max-steps", "10"]) == 0
    first = capsys.readouterr().out
    assert (out / "adapter_config.json").exists() and (out / "last" / "adapter_config.json").exists()
    meta = json.loads((out / "paper_meta.json").read_text())
    assert meta["step"] == 10 and meta["system_prompt"].startswith("You are Paper")
    assert "0 too long" in first

    assert train.main(["--config", cfgp, "--cpu", "--resume"]) == 0
    second = capsys.readouterr().out
    assert "resumed at step 10" in second
    seq = losses(first) + losses(second)
    # Only the adapters train and the base is random, so the loss cannot fall far; it must fall steadily.
    assert len(seq) > 30 and sum(seq[-5:]) / 5 < sum(seq[:5]) / 5 - 0.5, seq
    final = json.loads((out / "paper_meta.json").read_text())
    assert final["step"] > 10 and final["best_val_loss"] <= meta["best_val_loss"]

    from safetensors.torch import load_file
    weights = load_file(out / "adapter_model.safetensors")
    lora_b = [v for k, v in weights.items() if "lora_B" in k]
    assert lora_b and all(float(v.abs().sum()) > 0 for v in lora_b)  # B starts at zero: gradients reached it

    assert export.main(["--config", cfgp]) == 0
    merged = project / "out" / "paper-merged"
    assert (merged / "config.json").exists() and "FROM ./paper-Q4_K_M.gguf" in (project / "out" / "Modelfile").read_text()

    cfg = load(cfgp)
    for backend in (infer.HFBackend(str(merged), cpu=True),
                    infer.HFBackend(cfg["student"]["base_model"], str(out), cpu=True)):
        texts = backend.ask(cfg["system_prompt"], ["What is 2+2?", "Name a colour."], 2,
                            temperature=0.6, top_p=0.95, max_new_tokens=40, think_budget=16)
        assert len(texts) == 2 and all(len(t) == 2 and all(isinstance(x, str) for x in t) for t in texts)
        assert all("<|im_end|>" not in x and "<|endoftext|>" not in x for t in texts for x in t)
        res = infer.solve(backend, cfg["system_prompt"], "What is 2+2?", cfg["inference"], "vote")
        assert res["samples"] in (2, 3) and isinstance(res["answer"], str)
    # Merged weights must compute the same thing as base + adapter.
    m = infer.HFBackend(str(merged), cpu=True)
    b = infer.HFBackend(cfg["student"]["base_model"], str(out), cpu=True)
    ids = m.tok(infer.prompt_text(m.tok, cfg["system_prompt"], "What is 2+2?"), return_tensors="pt",
                add_special_tokens=False)
    with torch.no_grad():
        diff = (m.model(**ids).logits - b.model(**ids).logits).abs().max().item()
        untrained = transformers.AutoModelForCausalLM.from_pretrained(cfg["student"]["base_model"])
        moved = (m.model(**ids).logits - untrained(**ids).logits).abs().max().item()
    assert diff < 1e-3 < moved, (diff, moved)  # same as the adapter, and different from the untrained base
    greedy = dict(temperature=0.0, top_p=1.0, max_new_tokens=24, think_budget=12)
    assert m.ask(cfg["system_prompt"], ["What is 2+2?"], 1, **greedy) == m.ask(cfg["system_prompt"], ["What is 2+2?"], 1, **greedy)


def test_live_chat_output_and_pausing_training(project, capsys, monkeypatch):
    cfgp = str(project / "config.yaml")
    out = project / "out" / "paper-lora"
    if not (out / "adapter_config.json").exists():
        pytest.skip("needs the training test to have run")
    cfg = load(cfgp)
    backend = infer.HFBackend(cfg["student"]["base_model"], str(out), cpu=True)
    capsys.readouterr()
    prompt = infer.prompt_text(backend.tok, cfg["system_prompt"], "What is 2+2?")
    gen = dict(temperature=0.0, top_p=1.0, max_new_tokens=40, think_budget=16)
    silent = backend.complete([prompt], **gen)
    assert capsys.readouterr().out == ""                       # nothing is printed unless asked
    shown = backend.complete([prompt], show=True, **gen)
    live = capsys.readouterr().out
    # same reply either way; the live view starts with the reasoning itself, not its opening tag
    assert shown == silent and len(live.strip()) > 10 and not live.lstrip().startswith("<think>")
    assert "First I restate" in live

    # Ctrl+C part-way through training saves a checkpoint and the run can be resumed
    shutil.rmtree(out)
    real, calls = train.collate, {"n": 0}

    def interrupted(*a, **k):
        calls["n"] += 1
        if calls["n"] == 9:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(train, "collate", interrupted)
    assert train.main(["--config", cfgp, "--cpu"]) == 130
    text = capsys.readouterr().out
    assert "paused" in text and (out / "last" / "adapter_config.json").exists()
    paused_at = json.loads((out / "paper_meta.json").read_text())["step"]
    assert paused_at == 4                                      # eight batches, two per step
    monkeypatch.setattr(train, "collate", real)
    assert train.main(["--config", cfgp, "--cpu", "--resume", "--max-steps", "8"]) == 0
    assert f"resumed at step {paused_at}" in capsys.readouterr().out
    assert json.loads((out / "paper_meta.json").read_text())["step"] == 8

    # a batch that runs out of GPU memory is retried once after freeing memory, and training carries on
    real_loss, fails = train.chunked_loss, {"n": 0}

    def fails_once(*a, **k):
        if k.get("backward", a[3] if len(a) > 3 else True) and fails["n"] == 0:
            fails["n"] += 1
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return real_loss(*a, **k)

    monkeypatch.setattr(train, "chunked_loss", fails_once)
    assert train.main(["--config", cfgp, "--cpu", "--resume", "--max-steps", "10"]) == 0
    text = capsys.readouterr().out
    assert fails["n"] == 1 and "left out one batch" not in text
    assert json.loads((out / "paper_meta.json").read_text())["step"] == 10

    # a batch that fails twice is left out with a notice; if that keeps happening the run stops with advice
    def never_fits(*a, **k):
        if k.get("backward", a[3] if len(a) > 3 else True):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return real_loss(*a, **k)

    monkeypatch.setattr(train, "chunked_loss", never_fits)
    with pytest.raises(SystemExit, match="Out of GPU memory"):
        train.main(["--config", cfgp, "--cpu", "--resume", "--max-steps", "14"])
    assert "left out one batch" in capsys.readouterr().out
    monkeypatch.setattr(train, "chunked_loss", real_loss)


def test_sliced_loss_matches_the_standard_loss_and_gradients(project):
    """The memory-light loss must be the same number, with the same gradients, as the ordinary one."""
    cfg = load(str(project / "config.yaml"))
    tok = transformers.AutoTokenizer.from_pretrained(cfg["student"]["base_model"])
    rows = read_jsonl(cfg["data_dir"] / "paper_train.jsonl")[:5]
    examples, _ = train.encode_all(tok, rows, 512)
    batch = train.collate(examples, tok.pad_token_id, "cpu")
    torch.manual_seed(0)
    model = train.add_lora(train.load_base(cfg["student"]["base_model"], False, "cpu"),
                           {"lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.0}, False)
    for name, prm in model.named_parameters():  # LoRA's B matrices start at zero; give them values so
        if "lora_B" in name:                     # that every adapter weight receives a real gradient
            torch.nn.init.normal_(prm, std=0.05)
    model.train()
    params = {n: p_ for n, p_ in model.named_parameters() if p_.requires_grad}

    standard = model(**batch).loss
    (standard * 0.25).backward()
    want = {n: p_.grad.clone() for n, p_ in params.items()}
    model.zero_grad(set_to_none=True)

    for chunk in (7, 512):  # several slices, and one slice
        got_loss = train.chunked_loss(model, batch, scale=0.25, backward=True, chunk=chunk)
        assert abs(got_loss - standard.item()) < 1e-4, (got_loss, standard.item())
        for n, p_ in params.items():
            assert torch.allclose(p_.grad, want[n], rtol=2e-3, atol=1e-6), n
        model.zero_grad(set_to_none=True)
    with torch.no_grad():
        assert abs(train.chunked_loss(model, batch, backward=False, chunk=50) - standard.item()) < 1e-4
    assert all(p_.grad is None for p_ in params.values())  # the no-gradient call left nothing behind


def test_evaluate_runs_and_saves(project, capsys):
    cfgp = str(project / "config.yaml")
    if not (project / "out" / "paper-lora" / "adapter_config.json").exists():
        pytest.skip("needs the training test to have run")
    assert evaluate.main(["--config", cfgp, "--cpu", "--n", "6", "--batch", "3"]) == 0
    saved = list((project / "out").glob("eval_*_procedural_single.json"))
    assert len(saved) == 1
    res = json.loads(saved[0].read_text())
    assert res["overall"]["n"] == 6 and 0.0 <= res["overall"]["accuracy"] <= 1.0
    assert evaluate.main(["--config", cfgp, "--summary"]) == 0
    assert "accuracy" in capsys.readouterr().out
