"""Build the distillation dataset.

    python -m paper_distill.build prepare   # generate problems (no teacher needed)
    python -m paper_distill.build author    # the teacher writes new questions
    python -m paper_distill.build solve     # the teacher solves everything; wrong or poor traces are rejected
    python -m paper_distill.build select    # dedupe, balance, split, write the final files
    python -m paper_distill.build audit     # validate the final files
    python -m paper_distill.build all       # the five stages in order
    python -m paper_distill.build pilot     # 60 teacher calls, then a time and yield estimate

Every stage is resumable: stop it at any point and run it again.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from . import seeds as seedlib
from .config import append_jsonl, load, read_jsonl, write_jsonl
from .dedup import _shingles, near_duplicates
from .quality import trace_problems
from .schema import make_record, validate_record
from .teacher import Teacher, solve_messages
from .topics import TOPICS, author_prompt, parse_authored
from .verify import check_answer, extract_final_answer

SOURCES = ("procedural", "teacher_authored", "public")


def paths(cfg: dict) -> dict:
    d = cfg["data_dir"]
    return {"procedural": d / "seeds_procedural.jsonl", "teacher_authored": d / "seeds_authored.jsonl",
            "public": d / "seeds_public.jsonl", "eval": d / "eval_procedural.jsonl", "traces": d / "traces.jsonl",
            "train": d / "paper_train.jsonl", "val": d / "paper_val.jsonl", "report": d / "report.json"}


def quotas(cfg: dict) -> dict[str, int]:
    """How many final records each source should contribute."""
    ds = cfg["dataset"]
    return {s: round(ds["target_size"] * float(ds["mix"].get(s, 0))) for s in SOURCES}


def source_of(seed: dict) -> str:
    return "public" if seed["source"].startswith("public:") else seed["source"]


# ---------------------------------------------------------------- prepare

def prepare(cfg: dict) -> None:
    ds, p = cfg["dataset"], paths(cfg)
    q = quotas(cfg)
    if not p["eval"].exists():
        write_jsonl(p["eval"], seedlib.generate(ds["eval_holdout"], ds["seed"] + 1))
    held_out = {r["id"] for r in read_jsonl(p["eval"])}
    want = math.ceil(q["procedural"] * ds["oversample"])
    have = read_jsonl(p["procedural"])
    if ds["direct_share"] > 0 and not cfg["teacher"].get("direct_body"):
        raise SystemExit("dataset.direct_share > 0 needs teacher.direct_body (how to switch the teacher's "
                         "reasoning off). deepseek-r1 cannot do this; qwen3 can.")
    if len(have) < want:
        rows = seedlib.generate(want, ds["seed"], exclude=held_out)
        rng = random.Random(ds["seed"])
        for r in rows:  # a share of the easy problems is answered without reasoning, if the teacher can do that
            easy = r["difficulty"] <= 2
            r["mode"] = "direct" if easy and rng.random() < ds["direct_share"] * 2.5 else "think"
        write_jsonl(p["procedural"], rows)
        have = rows
    print(f"prepare: {len(have)} procedural problems, {len(held_out)} held out for evaluation")
    if q["public"] and not p["public"].exists():
        from .seeds.public import gsm8k_train
        rows = [r for r in gsm8k_train(math.ceil(q["public"] * ds["oversample"]), ds["seed"])]
        write_jsonl(p["public"], rows)
        print(f"prepare: {len(rows)} public problems (GSM8K train, MIT)")


# ---------------------------------------------------------------- author

def author(cfg: dict, teacher=None, limit: int | None = None) -> None:
    ds, au, p = cfg["dataset"], cfg["authoring"], paths(cfg)
    want = math.ceil(quotas(cfg)["teacher_authored"] * ds["oversample"])
    if want == 0:
        return
    teacher = teacher or Teacher(cfg["teacher"])
    rows = read_jsonl(p["teacher_authored"])
    keys = {r["id"] for r in rows}
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_topic[r["generator"]].append({**r, "_sh": _shingles(r["question"], 3)})
    cells = [(t, d) for d in (1, 2, 3, 4, 5) for t in TOPICS]
    weights = dict(zip((1, 2, 3, 4, 5), seedlib.DIFFICULTY_WEIGHTS))
    per_cell = {(t[1], d): math.ceil(want * weights[d] / len(TOPICS)) for t, d in cells}
    have = Counter((r["generator"], r["difficulty"]) for r in rows)
    calls, misses = 0, Counter()
    print(f"author: have {len(rows)} of {want} questions")
    while len(rows) < want:
        open_cells = [(t, d) for t, d in cells if have[(t[1], d)] < per_cell[(t[1], d)] and misses[(t[1], d)] < 4]
        if not open_cells or (limit is not None and calls >= limit):
            break
        (domain, skill, brief), d = min(open_cells, key=lambda c: have[(c[0][1], c[1])] / per_cell[(c[0][1], c[1])])
        recent = [r["question"] for r in by_topic[skill]]
        msg = [{"role": "user", "content": author_prompt(brief, d, au["batch_size"], recent)}]
        reply = teacher.chat(msg, max_tokens=au["max_tokens"])
        calls += 1
        added = 0
        for item in parse_authored(reply.content):
            key = seedlib.question_key(item["question"])
            sh = _shingles(item["question"], 3)
            similar = any(len(sh & o["_sh"]) / len(sh | o["_sh"]) >= au["near_duplicate"] for o in by_topic[skill])
            if key in keys or similar:
                continue
            row = {"id": key, "source": "teacher_authored", "generator": skill, "domain": domain, "skill": skill,
                   "difficulty": d, "question": item["question"], "ground_truth": item["answer"],
                   "answer_type": item["answer_type"], "meta": {}, "mode": "think"}
            append_jsonl(p["teacher_authored"], row)
            keys.add(key)
            rows.append(row)
            by_topic[skill].append({**row, "_sh": sh})
            have[(skill, d)] += 1
            added += 1
        misses[(skill, d)] += added == 0
        if calls % 10 == 0:
            print(f"author: {len(rows)}/{want} questions after {calls} calls", flush=True)
    print(f"author: {len(rows)} questions written by the teacher ({calls} calls this run)")


# ---------------------------------------------------------------- solve

def judge(seed: dict, reasoning: str, content: str, finish_reason, quality: dict) -> tuple[bool, list[str], str | None]:
    """Decide whether a teacher trace is kept. Returns (keep, reasons_if_not, final_answer)."""
    mode = seed.get("mode", "think")
    reasons = trace_problems(reasoning, content, finish_reason, quality, mode)
    final = extract_final_answer(content, strict=True)
    if final is None:
        reasons.append("no_final_answer_line")
    elif not check_answer(seed["answer_type"], seed["ground_truth"], final, seed.get("meta")):
        reasons.append("wrong_answer")
    return not reasons, reasons, final


class Scheduler:
    """Hands out the next question so that accepted traces stay balanced across generators.

    Harder generators are rejected more often; without this the kept set would drift
    toward whatever the teacher finds easy.
    """

    def __init__(self, seeds: list[dict], traces: list[dict], quota: dict[str, int], max_attempts: int):
        self.lock = threading.Lock()
        self.quota, self.max_attempts = quota, max_attempts
        self.attempts = Counter(t["seed_id"] for t in traces)
        accepted_ids = {t["seed_id"] for t in traces if t["ok"]}
        self.acc_src, self.acc_grp = Counter(), Counter()
        self.fly_src, self.fly_grp = Counter(), Counter()
        self.queues: dict[tuple[str, str], deque] = defaultdict(deque)
        for s in seeds:
            g = (source_of(s), s["generator"])
            if s["id"] in accepted_ids:
                self.acc_src[g[0]] += 1
                self.acc_grp[g] += 1
            elif self.attempts[s["id"]] < max_attempts:
                self.queues[g].append(s)

    def next(self):
        with self.lock:
            open_groups = [g for g, q in self.queues.items()
                           if q and self.acc_src[g[0]] + self.fly_src[g[0]] < self.quota[g[0]]]
            if not open_groups:
                return None
            g = min(open_groups, key=lambda g: (self.acc_grp[g] + self.fly_grp[g], g))
            seed = self.queues[g].popleft()
            self.fly_src[g[0]] += 1
            self.fly_grp[g] += 1
            return seed

    def done(self, seed: dict, ok: bool) -> None:
        with self.lock:
            g = (source_of(seed), seed["generator"])
            self.fly_src[g[0]] -= 1
            self.fly_grp[g] -= 1
            self.attempts[seed["id"]] += 1
            if ok:
                self.acc_src[g[0]] += 1
                self.acc_grp[g] += 1
            elif self.attempts[seed["id"]] < self.max_attempts:
                self.queues[g].append(seed)

    def release(self, seed: dict) -> None:
        """Put back a question that was handed out but never sent to the teacher."""
        with self.lock:
            g = (source_of(seed), seed["generator"])
            self.fly_src[g[0]] -= 1
            self.fly_grp[g] -= 1
            self.queues[g].appendleft(seed)

    def in_flight(self) -> int:
        with self.lock:
            return sum(self.fly_src.values())


def solve(cfg: dict, teacher=None, limit: int | None = None) -> dict:
    ds, tc, p = cfg["dataset"], cfg["teacher"], paths(cfg)
    teacher = teacher or Teacher(tc)
    seeds = [s for src in SOURCES for s in read_jsonl(p[src])]
    traces = read_jsonl(p["traces"])
    # ask for a little more than the final quota, to cover what select() later drops
    quota = {s: math.ceil(n * 1.06) for s, n in quotas(cfg).items()}
    sched = Scheduler(seeds, traces, quota, ds["max_attempts"])
    stats = {"calls": 0, "kept": 0, "tokens": 0, "reasons": Counter()}
    lock, t0, stop = threading.Lock(), time.time(), threading.Event()
    start_kept = sum(sched.acc_src.values())
    print(f"solve: {start_kept} traces already kept; target {sum(quota.values())}", flush=True)

    def take() -> bool:
        with lock:
            if limit is not None and stats["calls"] >= limit:
                return False
            stats["calls"] += 1
            return True

    def worker() -> None:
        while not stop.is_set():
            seed = sched.next()
            if seed is None:
                if sched.in_flight() == 0:
                    return
                time.sleep(0.2)
                continue
            if not take():
                sched.release(seed)
                stop.set()
                return
            direct = seed.get("mode") == "direct"
            try:
                reply = teacher.chat(solve_messages(cfg["instruction"], seed["question"], tc["system_in_user"]),
                                     direct=direct)
            except BaseException:
                sched.release(seed)
                stop.set()  # the teacher is down: let the other workers finish and report the error
                raise
            reasoning = "" if direct else reply.reasoning
            ok, reasons, final = judge(seed, reasoning, reply.content, reply.finish_reason, cfg["quality"])
            append_jsonl(p["traces"], {
                "seed_id": seed["id"], "ok": ok, "reasons": reasons, "final_answer": final,
                "reasoning": reasoning, "content": reply.content, "finish_reason": reply.finish_reason,
                "completion_tokens": reply.completion_tokens, "seconds": round(reply.seconds, 2),
                "model": tc["model"], "temperature": tc["temperature"]})
            sched.done(seed, ok)
            with lock:
                stats["kept"] += ok
                stats["tokens"] += reply.completion_tokens or 0
                stats["reasons"].update(reasons)
                n = stats["calls"]
                if n % 20 == 0:
                    el = time.time() - t0
                    rate = stats["kept"] / n
                    left = sum(quota.values()) - start_kept - stats["kept"]
                    eta = left / max(rate, 1e-6) * (el / n) / 3600
                    print(f"solve: {n} calls, kept {stats['kept']} ({rate:.0%}), "
                          f"{stats['tokens'] / el:.0f} tok/s, about {eta:.1f} h to go", flush=True)

    with ThreadPoolExecutor(max_workers=tc["concurrency"]) as pool:
        for f in [pool.submit(worker) for _ in range(tc["concurrency"])]:
            f.result()
    stats["seconds"] = time.time() - t0
    stats["remaining"] = max(0, sum(quota.values()) - start_kept - stats["kept"])
    print(f"solve: {stats['calls']} calls this run, {stats['kept']} kept. "
          f"Rejections: {dict(stats['reasons'].most_common())}")
    if stats["remaining"] and limit is None:
        print(f"solve: ran out of questions {stats['remaining']} short of target. "
              "Raise dataset.oversample and run prepare, author and solve again.")
    return stats


# ---------------------------------------------------------------- select

def select(cfg: dict) -> dict:
    ds, tc, p = cfg["dataset"], cfg["teacher"], paths(cfg)
    rng = random.Random(ds["seed"])
    seeds = {s["id"]: s for src in SOURCES for s in read_jsonl(p[src])}
    held_out = {r["id"] for r in read_jsonl(p["eval"])}
    pool: dict[str, dict] = {}
    dropped = Counter()
    for t in read_jsonl(p["traces"]):
        seed = seeds.get(t["seed_id"])
        if not (t["ok"] and seed) or seed["id"] in pool or seed["id"] in held_out:
            continue
        method = "ground_truth" if seed["source"] != "teacher_authored" else "teacher_agreement"
        rec = make_record(seed=seed, system=cfg["system_prompt"], reasoning=t["reasoning"], answer=t["content"],
                          final_answer=t["final_answer"], mode=seed.get("mode", "think"), method=method,
                          reference=seed["ground_truth"],
                          teacher={"model": t["model"], "license": tc["license"], "temperature": t["temperature"]},
                          completion_tokens=t["completion_tokens"])
        if validate_record(rec):
            dropped["invalid_record"] += 1
        else:
            pool[seed["id"]] = rec
    pool_size = len(pool) + dropped["invalid_record"]
    authored = [r for r in pool.values() if r["source"] == "teacher_authored"]
    for i in near_duplicates([r["question"] for r in authored], cfg["authoring"]["near_duplicate"]):
        del pool[authored[i]["id"]]
        dropped["near_duplicate"] += 1

    # Within each skill, when there is a surplus, let go of the longest traces first.
    by_src: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in pool.values():
        by_src[source_of(r)][r["skill"]].append(r)
    q = quotas(cfg)
    chosen: list[dict] = []
    for src in SOURCES:
        groups = by_src.get(src, {})
        for g in groups.values():
            rng.shuffle(g)
            limit = _p80(g)
            g.sort(key=lambda r: r["stats"]["reasoning_words"] > limit)  # stable sort: longest fifth goes last
        picked, i = [], 0
        while len(picked) < q[src] and any(i < len(g) for g in groups.values()):
            for k in sorted(groups):  # round-robin keeps skills balanced
                if i < len(groups[k]) and len(picked) < q[src]:
                    picked.append(groups[k][i])
            i += 1
        chosen += picked
    # Top up from whatever verified traces are left if one source came in short.
    short = ds["target_size"] - len(chosen)
    if short > 0:
        ids = {r["id"] for r in chosen}
        spare = [r for r in pool.values() if r["id"] not in ids]
        rng.shuffle(spare)
        chosen += spare[:short]
    # No single domain may dominate.
    cap = math.ceil(ds["max_domain_share"] * max(len(chosen), 1))
    seen, capped = Counter(), []
    rng.shuffle(chosen)
    for r in chosen:
        if seen[r["domain"]] < cap:
            seen[r["domain"]] += 1
            capped.append(r)
        else:
            dropped["domain_cap"] += 1
    capped.sort(key=lambda r: r["id"])
    rng.shuffle(capped)
    n_val = max(1, round(len(capped) * ds["val_fraction"])) if len(capped) > 1 else 0
    val, train = capped[:n_val], capped[n_val:]
    write_jsonl(p["train"], train)
    write_jsonl(p["val"], val)
    report = summarise(train + val)
    report.update({"train": len(train), "val": len(val), "verified_pool": pool_size,
                   "dropped": dict(dropped)})
    import json
    p["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"select: wrote {len(train)} train + {len(val)} validation records from a verified pool of "
          f"{report['verified_pool']}")
    return report


def _p80(group: list[dict]) -> int:
    lens = sorted(r["stats"]["reasoning_words"] for r in group)
    return lens[min(len(lens) - 1, int(0.8 * len(lens)))]


def summarise(rows: list[dict]) -> dict:
    words = sorted(r["stats"]["reasoning_words"] for r in rows) or [0]
    pct = lambda f: words[min(len(words) - 1, int(f * len(words)))]  # noqa: E731
    return {"total": len(rows),
            "by_source": dict(Counter(r["source"] for r in rows)),
            "by_domain": dict(Counter(r["domain"] for r in rows).most_common()),
            "by_difficulty": dict(sorted(Counter(r["difficulty"] for r in rows).items())),
            "by_mode": dict(Counter(r["mode"] for r in rows)),
            "by_verification": dict(Counter(r["verification"]["method"] for r in rows)),
            "reasoning_words": {"median": pct(0.5), "p90": pct(0.9), "max": words[-1]}}


# ---------------------------------------------------------------- audit

def audit(cfg: dict) -> bool:
    ds, p = cfg["dataset"], paths(cfg)
    train, val = read_jsonl(p["train"]), read_jsonl(p["val"])
    rows = train + val
    problems: list[str] = []
    for i, r in enumerate(rows):
        for e in validate_record(r):
            problems.append(f"record {i} ({r.get('id', '?')}): {e}")
    ids = Counter(r.get("id") for r in rows)
    problems += [f"duplicate id {k}" for k, n in ids.items() if n > 1]
    qs = Counter(seedlib.question_key(r["question"]) for r in rows if isinstance(r.get("question"), str))
    problems += [f"duplicate question {k}" for k, n in qs.items() if n > 1]
    leaked = set(ids) & {r["id"] for r in read_jsonl(p["eval"])}
    problems += [f"evaluation problem {k} is in the training data" for k in leaked]
    if len({r.get("system") for r in rows}) > 1:
        problems.append("records do not all share one system prompt")
    if len(rows) < ds["min_size"]:
        problems.append(f"only {len(rows)} records; dataset.min_size is {ds['min_size']}")
    rep = summarise(rows) if rows else {"total": 0}
    print(f"audit: {len(train)} train + {len(val)} validation records")
    for k in ("by_source", "by_verification", "by_mode", "by_difficulty", "by_domain", "reasoning_words"):
        if k in rep:
            print(f"  {k}: {rep[k]}")
    if problems:
        print(f"audit: FAILED with {len(problems)} problem(s):")
        for line in problems[:20]:
            print("  -", line)
        return False
    print("audit: passed. Every record is valid, unique, and absent from the evaluation hold-out.")
    return True


# ---------------------------------------------------------------- pilot

def pilot(cfg: dict, teacher=None, calls: int = 60) -> None:
    prepare(cfg)
    stats = solve(cfg, teacher, limit=calls)
    if not stats["calls"]:
        print("pilot: nothing left to solve")
        return
    rate = stats["kept"] / stats["calls"]
    per_call = stats["seconds"] / stats["calls"]
    need = sum(quotas(cfg).values())
    hours = need / max(rate, 0.01) * per_call / 3600
    print(f"pilot: {rate:.0%} of traces kept, {per_call:.1f} s per call at concurrency "
          f"{cfg['teacher']['concurrency']}.")
    print(f"pilot: reaching {need} records would take roughly {hours:.0f} hours of solving, plus question "
          "writing. Teacher-written questions are usually kept less often than generated ones.")
    top = stats["reasons"].most_common(1)
    if top and top[0][0] == "reasoning_too_short" and top[0][1] > stats["calls"] / 2:
        print("pilot: most replies came back with no reasoning. The server is probably not returning the model's "
              "thinking. For Ollama, try `extra_body: {reasoning_effort: medium}` under teacher in config.yaml.")
    if top and top[0][0] == "truncated" and top[0][1] > stats["calls"] / 2:
        print("pilot: most replies were cut off. Check the server's context window (see the README, step 1).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "author", "solve", "select", "audit", "all", "pilot"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many teacher calls")
    a = ap.parse_args(argv)
    cfg = load(a.config)
    if a.stage == "prepare":
        prepare(cfg)
    elif a.stage == "author":
        author(cfg, limit=a.limit)
    elif a.stage == "solve":
        solve(cfg, limit=a.limit)
    elif a.stage == "select":
        select(cfg)
    elif a.stage == "pilot":
        pilot(cfg, calls=a.limit or 60)
    elif a.stage == "all":
        prepare(cfg)
        author(cfg)
        solve(cfg)
        select(cfg)
    if a.stage in ("audit", "all"):
        return 0 if audit(cfg) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
