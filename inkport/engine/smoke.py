"""Offline smoke-test fixtures: a tiny chat model and synthetic small-business data.

`build_smoke_model` trains a byte-level BPE tokenizer and warm-starts a ~2M-parameter Llama
architecture on *generic* chat skills (copying JSON fields and numbers, replying with a word).
It never sees the expense-categorization task, so its baseline on that task is ~0%, while a
LoRA adapter can learn the task in a couple of CPU minutes. This makes the whole pipeline
(ingest -> train -> gate -> serve) testable without downloads or a GPU.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from engine import events
from engine.suites import general

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)
SPECIAL = ["<unk>", "<pad>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]

CATEGORIES = {
    "fuel": ["Shell", "Chevron", "Exxon", "BP", "Valero"],
    "meals": ["Chipotle", "Panera", "Subway", "Starbucks", "Denny's"],
    "software": ["Adobe", "GitHub", "Zoom", "Slack", "Dropbox"],
    "office_supplies": ["Staples", "OfficeMax", "Uline", "Office Depot"],
    "travel": ["Delta", "Hilton", "Marriott", "Uber", "Amtrak"],
}
PURPOSES = ["for the delivery van", "client lunch", "team subscription", "printer paper", "trip to Dallas", "monthly plan", "job site", "supplies run"]
PATTERNS = ["Paid ${a} at {v} {p}", "{v} receipt ${a} {p}", "Card charge: {v} ${a}", "{v} - ${a} ({p})", "Reimburse ${a} spent at {v}"]
WORDS = ["apple", "river", "blue", "stone", "cloud", "maple", "north", "pixel", "garden", "copper", "amount", "category", "vendor", "total", "color", "name", "status"]


def expense_rows(n: int = 320, seed: int = 3) -> list[dict]:
    rng = random.Random(seed)
    rows, seen = [], set()
    while len(rows) < n:
        cat = rng.choice(sorted(CATEGORIES))
        vendor = rng.choice(CATEGORIES[cat])
        amount = rng.randint(3, 480)  # whole dollars keep the 1.4M-parameter smoke model's copy task learnable
        text = rng.choice(PATTERNS).format(a=str(amount), v=vendor, p=rng.choice(PURPOSES))
        if text in seen:
            continue
        seen.add(text)
        rows.append({"description": text, "category": cat, "amount": amount})
    return rows


def write_smoke_data(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = expense_rows()
    with open(out / "expenses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["description", "category", "amount"])
        w.writeheader()
        w.writerows(rows)
    (out / "policy.md").write_text("# Expense policy\n\nFuel is for company vehicles only. Meals with clients are business meals.\n", encoding="utf-8")
    rng = random.Random(5)
    with open(out / "orders.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["order_id", "customer_id", "region", "product", "quantity", "total"])
        w.writeheader()
        for i in range(60):
            q = rng.randint(1, 9)
            w.writerow({"order_id": i + 1, "customer_id": rng.randint(1, 8), "region": rng.choice(["North", "South", "East", "West"]), "product": rng.choice(["widget", "gadget", "gizmo"]), "quantity": q, "total": round(q * rng.uniform(5, 40), 2)})
    with open(out / "customers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["customer_id", "name", "segment"])
        w.writeheader()
        for i in range(8):
            w.writerow({"customer_id": i + 1, "name": f"Customer {i + 1}", "segment": rng.choice(["retail", "wholesale"])})
    (out / "helpers.py").write_text(
        'def add_tax(amount, rate):\n    """Return amount with sales tax, rounded to cents.\n\n    >>> add_tax(100, 0.07)\n    107.0\n    >>> add_tax(19.99, 0.0825)\n    21.64\n    """\n    return round(amount * (1 + rate), 2)\n\n\n'
        'def initials(name):\n    """Return uppercase initials.\n\n    >>> initials("ada lovelace")\n    \'AL\'\n    >>> initials("Grace Brewster Hopper")\n    \'GBH\'\n    """\n    return "".join(p[0].upper() for p in name.split())\n',
        encoding="utf-8",
    )
    (out / "lease.md").write_text(
        "# Residential Lease\n\nThe monthly rent is $1,450 and is due on the first day of each month. "
        "A late fee of $75 applies if rent is more than 5 days late.\n\nThe security deposit of $2,000 will be returned within 30 days after the Tenant moves out. "
        "The lease term begins on March 1, 2026 and ends on February 28, 2027.\n\nThe Landlord must give 24 hours notice before entering the unit. "
        "Pets are allowed with a deposit of $300 per pet.\n",
        encoding="utf-8",
    )
    return {"dir": str(out), "files": sorted(p.name for p in out.iterdir())}


def _generic_examples(rng: random.Random, n: int) -> list[list[dict]]:
    """Generic skills only. The 'extract' family uses the real business-template prompt with
    unrelated words: copy the dollar amount and pick the allowed choice that literally appears
    in the text. The real task (vendor -> category, where the category never appears in the
    text) is never shown, so the base model scores ~0% on it."""
    from engine.templates import business_system  # noqa: WPS433

    out = []
    for _ in range(n):
        r = rng.random()
        msgs: list[dict] = []
        if r < 0.55:
            field = rng.choice(["color", "status", "kind", "label", "group"])
            choices = rng.sample(WORDS, 4)
            pick = rng.choice(choices)
            amount = rng.randint(3, 480)
            text = rng.choice(["Paid ${a} at {w} for {p}", "{w} receipt ${a} {p}", "Card charge: {w} ${a}", "{w} - ${a} ({p})", "Reimburse ${a} spent at {w}"]).format(a=amount, w=pick.capitalize() if rng.random() < 0.5 else pick, p=rng.choice(PURPOSES))
            rules = rng.sample([f"{field} must be one of {', '.join(choices)}", "amount must be positive", "amount must be at most 500", f"{field} is required"], rng.randint(0, 2))
            policy = rng.choice(["", "", "# Policy\n\nKeep receipts for every purchase. Personal items are not reimbursed.", "# Expense policy\n\nUse the company card for supplies. Submit receipts weekly."])
            msgs.append({"role": "system", "content": business_system([field, "amount"], {field: choices}, rules, policy)})
            q, a = text, json.dumps({field: pick, "amount": amount})
        elif r < 0.7:
            k = rng.choice(WORDS)
            v = rng.choice(WORDS) if rng.random() < 0.5 else rng.randint(1, 500)
            q, a = f"Return JSON with {k} set to {v}", json.dumps({k: v})
        elif r < 0.9:
            it = rng.choice(general.suite(60) + general.replay_prompts(64))
            q, a = it["messages"][0]["content"], it.get("reference") or "OK"
        else:
            k = rng.choice(WORDS)
            q, a = f"Reply with only the word: {k}", k
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        out.append(msgs)
    return out


def build_smoke_model(out: Path, steps: int = 800, seed: int = 0) -> dict:
    import torch  # noqa: WPS433
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: WPS433
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: WPS433
    from jinja2 import Template  # noqa: WPS433

    rng = random.Random(seed)
    torch.manual_seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    events.stage("smoke_model", "started", "Building a tiny offline test model")
    tmpl = Template(CHAT_TEMPLATE)
    convs = _generic_examples(rng, 4000)
    corpus = [tmpl.render(messages=c, add_generation_prompt=False) for c in convs]
    # Tokenizer coverage for the task vocabulary (text only; no labels are paired with inputs).
    corpus += [r["description"] for r in expense_rows(400, seed=11)] + list(CATEGORIES) + ["office_supplies", "description", "category"]
    bpe = Tokenizer(models.BPE(unk_token="<unk>"))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=1200, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, unk_token="<unk>", pad_token="<pad>", eos_token="<|end|>", additional_special_tokens=SPECIAL[2:])
    tok.chat_template = CHAT_TEMPLATE
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=192, intermediate_size=384, num_hidden_layers=3, num_attention_heads=6, num_key_value_heads=2, max_position_embeddings=1024, tie_word_embeddings=False, bos_token_id=None, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model = LlamaForCausalLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)

    def enc(c: list[dict]) -> tuple[list[int], list[int]]:
        prefix = tmpl.render(messages=c[:-1], add_generation_prompt=True)
        full = tmpl.render(messages=c, add_generation_prompt=False)
        p = tok(prefix, add_special_tokens=False)["input_ids"]
        f = tok(full, add_special_tokens=False)["input_ids"][:256]
        return f, [-100] * len(p) + f[len(p) :]

    data = [enc(c) for c in convs]
    prog = events.Progress("smoke_model", steps, 0, 100)
    model.train()
    for step in range(steps):
        batch = rng.sample(data, 32)
        L = max(len(i) for i, _ in batch)
        ids = torch.tensor([i + [tok.pad_token_id] * (L - len(i)) for i, _ in batch])
        lab = torch.tensor([lb[:L] + [-100] * (L - len(lb[:L])) for _, lb in batch])
        att = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i, _ in batch])
        for g in opt.param_groups:
            g["lr"] = 2e-3 * min(1.0, (step + 1) / 50)
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        prog.update(step + 1, {"loss": round(float(loss), 4)})
    model.eval()
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    (out / "barnraise_smoke.json").write_text(json.dumps({"steps": steps, "seed": seed, "params": sum(p.numel() for p in model.parameters())}))
    events.stage("smoke_model", "completed", f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    return {"path": str(out), "params": sum(p.numel() for p in model.parameters()), "final_loss": float(loss)}
