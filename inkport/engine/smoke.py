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


# A sample document long enough to hold a fair hidden exam: every sentence states a fact (an amount,
# a date, a length of time, a percentage or a name) that a quiz question can be made from.
LEASE = """# Residential Lease Agreement

## 1. Parties and premises

This lease is made on February 12, 2026 between the landlord, Harborview Property Group, and the tenant, Maria Okafor. The premises are Apartment 4B at 218 Linden Street, on the fourth floor of the building known as Linden Court. The apartment has two bedrooms and may be occupied by no more than 4 people. The building manager is Daniel Reyes, who can be reached at the office from Monday to Friday and answers messages within 2 days. The tenant received 3 keys and 2 entry fobs on the day the lease was signed.

## 2. Term

The lease term begins on March 1, 2026 and ends on February 28, 2027. The tenant may move in up to 2 days before the start date at no extra charge. If the tenant stays after the end date without a new lease, the tenancy continues month to month and the rent rises by 8%. Either party may end a month-to-month tenancy with 60 days of written notice. The tenant may not assign this lease during the first 6 months of the term.

## 3. Rent

The monthly rent is $1,450 and is due on the first day of each month. Rent for a partial first month is charged at $48 per day. The tenant may pay by bank transfer, or by card with a processing fee of 2.5%. Rent paid a full year in advance earns a discount of 3%. The landlord must give 90 days of notice before any rent increase. An increase may not be more than 5% in any twelve-month period.

## 4. Late payment

A late fee of $75 applies if rent is more than 5 days late. After 15 days, unpaid rent also accrues interest at 1.5% per month. A returned payment costs the tenant $35 each time. If rent is more than 30 days late, the landlord may begin eviction proceedings. Partial payments are applied first to fees and then to the oldest unpaid rent, as explained by the office of Harborview Property Group.

## 5. Security deposit

The security deposit is $2,000 and is held in a separate account at Lakeside Community Bank. The deposit earns interest of 1% per year, paid to the tenant when the lease ends. The deposit will be returned within 30 days after the tenant moves out. The landlord must send an itemized list of any deductions within 21 days. Normal wear, such as small nail holes, may not be deducted, and repainting may be charged only after fewer than 2 years of occupancy. A missing entry fob costs $45 to replace.

## 6. Utilities

The landlord pays for water and trash collection, and the tenant pays for electricity and internet. Heating is included from October 15 to April 30 each year. If the tenant's share of the water bill exceeds $60 in a month, the tenant pays the difference. The electricity account must be opened in the tenant's name within 7 days of moving in. The building's internet provider is Brightline Fiber, although the tenant may choose another company. Trash is collected every Tuesday and recycling every Friday, according to the notice from Daniel Reyes.

## 7. Repairs and maintenance

The tenant must report any leak or electrical fault within 24 hours of noticing it. The landlord will begin urgent repairs within 48 hours and other repairs within 10 days. The tenant pays for damage caused by the tenant or guests, up to the first $150 of each repair. Smoke detectors are tested by the landlord every 6 months, and the tenant must not remove their batteries. Air filters are replaced every 3 months at the landlord's expense. The emergency repair line is run by a contractor called Northgate Maintenance Services.

## 8. Entry by the landlord

The landlord must give 24 hours of notice before entering the apartment, except in an emergency. Visits to show the apartment to new tenants may take place only during the last 45 days of the lease. Routine inspections happen no more than 2 times per year and last less than 1 hour. The landlord may enter without notice if water is leaking into another apartment, as required by the rules of Linden Court.

## 9. Pets

Pets are allowed with a deposit of $300 per pet. No more than 2 pets may live in the apartment, and each must weigh under 25 kilograms. Monthly pet rent is $25 for a cat and $40 for a dog. Dogs must be kept on a lead in the shared areas, and no pet may be left alone for more than 12 hours. The pet deposit is returned within 30 days if there is no pet damage, as confirmed by Harborview Property Group.

## 10. Parking and storage

One parking space, number 17, is included in the rent. A second parking space may be rented for $85 per month. A storage unit in the basement costs $30 per month and measures 4 square meters. Visitors may park in the marked bays for up to 12 hours. A vehicle left in a visitor bay for more than 3 days may be towed at the owner's expense. Bicycles are kept in the rack beside the entrance on Linden Street.

## 11. Guests and subletting

A guest may stay for up to 14 days in any six-month period without the landlord's approval. Subletting requires written consent, and the landlord will answer a request within 10 days. A sublet may last no longer than 6 months. Short-term rentals of under 30 days are not allowed in the building. The tenant remains responsible for the rent during any sublet, as stated by the landlord, Harborview Property Group.

## 12. Noise and shared areas

Quiet hours run from 10 p.m. to 7 a.m. on all 7 days of the week. The laundry room is open for 14 hours each day, and a wash costs $2.50. The roof terrace may be reserved for private events for a fee of $50. A reservation must be made at least 5 days ahead with the building manager, Daniel Reyes. Smoking is not allowed within 8 meters of any door or window.

## 13. Insurance

The tenant must hold renter's insurance with liability cover of at least $100,000. Proof of insurance must be given to the landlord within 14 days of the start of the lease. The landlord's own policy, issued by Granite Mutual Insurance, covers the building but not the tenant's belongings. If the apartment cannot be lived in after a fire, rent stops until it is repaired, and either party may end the lease after 60 days.

## 14. Renewal and ending the lease

The landlord will offer a renewal at least 75 days before the lease ends. The tenant must accept or decline the offer within 30 days of receiving it. To end the lease early, the tenant must give 60 days of notice and pay a fee equal to 2 months of rent. The early-ending fee is waived if the tenant is called to military service or moves more than 80 kilometers away for work. A tenant who has lived in the apartment for more than 3 years receives a loyalty discount of 4% on renewal.

## 15. Moving out

The tenant must return all keys within 24 hours of the lease ending. A move-out inspection takes place within 3 days of the keys being returned. Professional carpet cleaning, if needed, is charged at $120. Items left behind are stored for 30 days and then disposed of. The lift may be reserved for moving for up to 4 hours, with a refundable booking deposit of $100.

## 16. Notices and disputes

Written notices to the landlord are sent to the office of Harborview Property Group at 40 Quay Road. A notice sent by post is treated as received 3 days after posting. Disputes are first referred to mediation at the Riverside Housing Mediation Service, and each party pays 50% of the cost. Mediation must begin within 20 days of a written request. This lease was signed on February 12, 2026 by both parties.

## 17. Building services

The front desk is staffed for 12 hours each day, and parcels are held for up to 10 days. A lost mailbox key costs $20 to replace. The gym on the ground floor is free for tenants, and a guest pass costs $8 per visit. The lift is serviced every 3 months by Summit Elevator Company. Window cleaning takes place 2 times per year, in April and October. The building's fire alarm is tested on the first Monday of each month and lasts about 5 minutes, according to the building manager, Daniel Reyes. Heating problems are answered within 4 hours during the winter months.

## 18. Alterations and decoration

The tenant may hang pictures, but may not paint the walls without written consent. A request to redecorate is answered within 14 days. Shelving fixed to a wall must be removed at the end of the lease, and each unfilled hole larger than a coin is charged at $15. The tenant may install a washing machine if a licensed plumber fits it, and must keep the invoice for 2 years. Satellite dishes are not allowed, and an unauthorized dish is removed at a cost of $90. Window boxes are allowed on the balcony up to a weight of 20 kilograms. Any improvement costing more than $500 needs the approval of Harborview Property Group.

## 19. Safety and access

The entry code is changed every 6 months, and tenants are told 7 days before each change. A replacement entry code after a lockout costs $60 outside office hours. Fire doors in the corridors must stay closed, and blocking one carries a charge of $40. Barbecues are allowed only on the roof terrace and must be put out within 2 hours of use. The basement is closed for cleaning for 3 days each August. Each apartment has one fire extinguisher, which is inspected every 12 months by Northgate Maintenance Services. A tenant who is away for more than 21 days must tell the building manager.

## 20. Fees at a glance

The application fee paid before signing was $55 and is not refundable. A copy of this lease costs $10, and a letter confirming the tenancy costs $25. Changing the name of a tenant on the lease costs $150. A second set of keys costs $30, and a replacement parking permit costs $12. Late return of the lift booking deposit form delays the refund by up to 14 days. The cleaning charge for the shared barbecue area is $35 if it is left dirty. All fees in this section may rise by no more than 3% per year, as agreed with the tenant, Maria Okafor.

## 21. Cleaning and waste

The corridors and stairs are cleaned 3 times per week by a company called Clearwater Cleaning Services. Bulky items such as furniture are collected on the last Thursday of each month, for a charge of $18 per item. Rubbish left in a corridor is removed at a cost of $25 per bag. The bin store is locked at night and reopens after 6 hours. Glass is recycled separately, and the glass container is emptied every 2 weeks. Cooking oil must not be poured into the sinks, and clearing a blocked drain caused by oil costs $130. The tenant must clean the balcony drain at least 4 times per year.

## 22. Absence and emergencies

During an absence of more than 21 days the heating must be left on at no less than 12 degrees. The landlord may check the apartment once every 14 days while the tenant is away. A burst pipe must be reported to the emergency line within 1 hour of being discovered. The emergency line promises to call back within 30 minutes at any time of day. If the apartment floods, the landlord pays for a hotel for up to 5 days at no more than $140 per night. The tenant's emergency contact is recorded as Samuel Okafor. Power cuts lasting more than 48 hours reduce the rent by 10% for that month.

## 23. Rent reviews and records

The rent is reviewed once a year, on January 15, and any change takes effect 90 days later. A rent statement is sent to the tenant every 3 months. Receipts for rent paid in cash are issued within 2 days. The landlord keeps payment records for 6 years after the lease ends. A dispute about the amount owed must be raised within 60 days of the statement. An overpayment of more than $20 is refunded within 10 days, and smaller amounts are credited to the next month. The annual service charge statement is prepared by an accountant, Priya Nair, and is available on request.
"""


# Questions customers ask a small shop, each asked in several ways, with the reply the shop wants.
# A sample for the "custom skill" goal: the replies are short, polite and always say thank you,
# so the two suggested rules ("no more than 50 words", 'must mention "thank you"') hold.
SUPPORT = [
    (["What time do you open on weekdays?", "When are you open Monday to Friday?", "What are your weekday opening hours?", "How late are you open on a weekday?", "Until what time can I come by on a Tuesday?"],
     "We are open from 8 am to 6 pm, Monday to Friday. Thank you for asking!"),
    (["Are you open on Saturday?", "What are your Saturday hours?", "Can I shop with you on a Saturday?", "When do you close on Saturdays?", "Do you open at the weekend?"],
     "Yes, we are open on Saturdays from 9 am to 4 pm, and closed on Sundays. Thank you for checking!"),
    (["Are you open on public holidays?", "Do you close on bank holidays?", "Will the shop be open on a holiday?", "What are your holiday hours?", "Can I visit on a public holiday?"],
     "We are closed on public holidays and reopen the next working day at 8 am. Thank you for your patience!"),
    (["How long do I have to return something?", "What is your return policy?", "Can I bring an item back?", "How many days do I have for a return?", "I changed my mind about a purchase, can I return it?"],
     "You can return unused items within 30 days with your receipt for a full refund. Thank you for shopping with us!"),
    (["Can I return something without a receipt?", "I lost my receipt, can I still return this?", "Do I need the receipt for a return?", "What if I have no proof of purchase?", "Is a return possible without the receipt?"],
     "Without a receipt we can offer store credit for the current price of the item. Thank you for understanding!"),
    (["Can I return a power tool I have used?", "Do you take back used tools?", "I used the drill once, can I return it?", "What is the return rule for opened power tools?", "Are used power tools refundable?"],
     "Used power tools can be exchanged within 14 days if they are faulty, but not refunded. Thank you for asking first!"),
    (["How much is delivery?", "What do you charge for shipping?", "Is there a delivery fee?", "What does it cost to have an order delivered?", "How much is shipping for a small order?"],
     "Delivery costs $6 for orders under $75 and is free above that. Thank you for your order!"),
    (["How long does delivery take?", "When will my order arrive?", "How fast do you ship?", "How many days does shipping take?", "What is the delivery time?"],
     "Orders placed before 2 pm ship the same day and usually arrive within 3 working days. Thank you for your patience!"),
    (["Do you deliver bulky items like timber?", "Can you deliver long lengths of wood?", "Do you ship heavy items?", "Is delivery available for sheet material?", "Can large orders be brought to my site?"],
     "Yes, bulky items are delivered by our own van within 25 miles for a flat $20. Thank you for asking!"),
    (["Can I collect my order from the shop?", "Do you offer click and collect?", "Is in-store pickup possible?", "Can I order online and pick it up?", "How does collection work?"],
     "Yes, online orders are ready to collect within 2 hours during opening times. Thank you for choosing collection!"),
    (["Which payment methods do you take?", "Can I pay by card?", "Do you accept contactless payments?", "How can I pay in the shop?", "Do you take cash?"],
     "We accept cash, all major cards and contactless payments in the shop and online. Thank you for asking!"),
    (["Do you offer trade accounts?", "Can my business open an account with you?", "Is there an account for contractors?", "How do I get a trade account?", "Do builders get an account?"],
     "Yes, trade customers can open an account with 30 days to pay and 10% off list prices. Thank you for your interest!"),
    (["Do you price match?", "Will you match a competitor's price?", "I found it cheaper elsewhere, can you match that?", "Is there a price promise?", "Do you beat other shops' prices?"],
     "We match the price of any local shop on identical items that are in stock. Thank you for giving us the chance!"),
    (["Do you cut keys?", "Can you copy a house key?", "Is there a key cutting service?", "How long does it take to cut a key?", "Can I get a spare key made?"],
     "Yes, we cut most house keys while you wait, usually in about 5 minutes. Thank you for asking!"),
    (["Can you mix paint to a colour?", "Do you offer paint mixing?", "Can you match a paint colour for me?", "I need a custom paint colour, can you help?", "Is colour matching available?"],
     "Yes, we mix paint to any colour code and can match a sample you bring in. Thank you for thinking of us!"),
    (["Do you cut timber to size?", "Can you cut wood for me?", "Is there a timber cutting service?", "Will you cut a sheet of plywood to my measurements?", "How much do you charge to cut wood?"],
     "The first 3 cuts on timber bought from us are free, and extra cuts cost $1 each. Thank you for asking!"),
    (["Do you sharpen tools?", "Can you sharpen my chisels?", "Is there a sharpening service?", "Can I get garden shears sharpened?", "How long does blade sharpening take?"],
     "Yes, we sharpen blades and garden tools, and they are ready within 2 working days. Thank you for your custom!"),
    (["Do you hire out tools?", "Can I rent a floor sander?", "Is tool hire available?", "How much is it to hire a tile cutter?", "Do you rent equipment by the day?"],
     "Yes, we hire tools by the day or the week, with a refundable deposit taken at pickup. Thank you for asking!"),
    (["What warranty do your tools have?", "How long is the guarantee on power tools?", "Is there a warranty on what I buy?", "What happens if a tool breaks?", "Are your tools guaranteed?"],
     "Power tools carry a 2 year warranty, and hand tools are guaranteed for 10 years. Thank you for your trust!"),
    (["Do you sell gift cards?", "Can I buy a voucher?", "Are gift vouchers available?", "I need a present, do you have gift cards?", "Can gift cards be used online?"],
     "Yes, gift cards are available from $10 and can be used in the shop and online. Thank you for the kind thought!"),
    (["Is there parking at the shop?", "Where can I park?", "Do you have a car park?", "Can I park a van outside?", "Is parking free?"],
     "There is free parking for 20 cars behind the shop, with space for vans. Thank you for visiting!"),
    (["Can you order an item you do not stock?", "Do you take special orders?", "Can you get in a part for me?", "I need something you do not have, can you order it?", "How long do special orders take?"],
     "Yes, we can order most items within 5 working days with a 20% deposit. Thank you for asking!"),
    (["Do you recycle old batteries?", "Can I drop off used batteries?", "Do you take old paint tins?", "Is there a recycling point in the shop?", "Where can I recycle light bulbs?"],
     "Yes, you can leave used batteries and light bulbs at the recycling point by the door. Thank you for recycling!"),
    (["How do I contact customer service?", "What is the best way to reach you?", "Do you have a phone number?", "Can I email you a question?", "Who do I speak to about an order?"],
     "You can call us during opening hours or email at any time, and we reply within one working day. Thank you for getting in touch!"),
    (["Do you have a loyalty scheme?", "Is there a rewards card?", "Do regular customers get a discount?", "How do I collect points?", "Do you offer a membership?"],
     "Yes, our free loyalty card gives you 1 point per dollar and $5 off for every 100 points. Thank you for your loyalty!"),
    (["Do you give advice on DIY projects?", "Can someone help me choose the right screws?", "Is there staff who can advise on plumbing?", "Can I ask for help planning a project?", "Do you offer free advice?"],
     "Yes, our staff are happy to advise on any project, free of charge, in the shop or by phone. Thank you for asking!"),
]


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
            w.writerow({"order_id": i + 1, "customer_id": rng.randint(1, 8), "region": rng.choice(["North", "South", "East", "West"]), "product": rng.choice(["widget", "gadget", "gizmo", "sprocket"]), "quantity": q, "total": round(q * rng.uniform(5, 40), 2)})
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
    (out / "lease.md").write_text(LEASE, encoding="utf-8")
    with open(out / "support.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "answer"])
        for questions, answer in SUPPORT:
            for question in questions:
                w.writerow([question, answer])
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
    (out / "inkport_smoke.json").write_text(json.dumps({"steps": steps, "seed": seed, "params": sum(p.numel() for p in model.parameters())}))
    events.stage("smoke_model", "completed", f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    return {"path": str(out), "params": sum(p.numel() for p in model.parameters()), "final_loss": float(loss)}
