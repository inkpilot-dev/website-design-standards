"""Build the distillation dataset.

    python -m paper_distill.build prepare   # generate problems (no teacher needed)
    python -m paper_distill.build author    # the teacher writes new questions
    python -m paper_distill.build solve     # the teacher solves everything; wrong or poor traces are rejected
    python -m paper_distill.build select    # dedupe, balance, split, write the final files
    python -m paper_distill.build audit     # validate the final files
    python -m paper_distill.build all       # the five stages in order
    python -m paper_distill.build pilot     # a short trial run, then a time and yield estimate
    python -m paper_distill.build stats     # keep rates so far, by source and difficulty
    python -m paper_distill.build sample    # print a couple of kept traces in full

Every stage is resumable: stop it at any point and run it again.
"""
from __future__ import annotations

import argparse
import math
import os
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


def run_workers(worker, n: int, stop: threading.Event, what: str) -> None:
    """Run `worker` on n threads until they all return. Ctrl+C sets `stop` and waits for the
    requests in progress; a second Ctrl+C exits at once. A worker's exception is re-raised."""
    pool = ThreadPoolExecutor(max_workers=n)
    futures = [pool.submit(worker) for _ in range(n)]
    try:
        while not all(f.done() for f in futures):
            time.sleep(0.3)  # a short sleep, not a blocking wait, so Ctrl+C is noticed on Windows too
    except KeyboardInterrupt:
        stop.set()
        print(f"{what}: stopping after the requests in progress (Ctrl+C again to quit at once; nothing is lost)",
              flush=True)
        try:
            while not all(f.done() for f in futures):
                time.sleep(0.3)
        except KeyboardInterrupt:
            os._exit(130)
        raise
    finally:
        pool.shutdown(wait=False)
    for f in futures:
        f.result()


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
    """Have the teacher write new questions, several requests at a time."""
    ds, au, p = cfg["dataset"], cfg["authoring"], paths(cfg)
    want = math.ceil(quotas(cfg)["teacher_authored"] * au.get("oversample", ds["oversample"]))
    if want == 0:
        return
    teacher = teacher or Teacher(cfg["teacher"])
    rows = read_jsonl(p["teacher_authored"])
    keys = {r["id"] for r in rows}
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_topic[r["generator"]].append({**r, "_sh": _shingles(r["question"], 3)})
    cells = [(t, d) for d in (1, 2, 3, 4, 5) for t in TOPICS]
    random.Random(ds["seed"]).shuffle(cells)  # so even a short run covers a mix of topics and difficulties
    weights = dict(zip((1, 2, 3, 4, 5), seedlib.DIFFICULTY_WEIGHTS))
    per_cell = {(t[1], d): math.ceil(want * weights[d] / len(TOPICS)) for t, d in cells}
    have = Counter((r["generator"], r["difficulty"]) for r in rows)
    misses, busy = Counter(), set()
    state = {"calls": 0}
    lock, stop = threading.Lock(), threading.Event()
    print(f"author: have {len(rows)} of {want} questions", flush=True)

    def pick():
        """The emptiest cell nobody is working on. None: finished. False: wait for a busy cell."""
        if len(rows) >= want or (limit is not None and state["calls"] >= limit):
            return None
        open_cells = [(t, d) for t, d in cells if have[(t[1], d)] < per_cell[(t[1], d)] and misses[(t[1], d)] < 4]
        free = [c for c in open_cells if (c[0][1], c[1]) not in busy]
        if not free:
            return False if open_cells else None
        topic, d = min(free, key=lambda c: have[(c[0][1], c[1])] / per_cell[(c[0][1], c[1])])
        busy.add((topic[1], d))
        state["calls"] += 1
        return topic, d, [r["question"] for r in by_topic[topic[1]]]

    def worker() -> None:
        while not stop.is_set():
            with lock:
                job = pick()
            if job is None:
                return
            if job is False:
                time.sleep(0.2)
                continue
            (domain, skill, brief), d, recent = job
            try:
                reply = teacher.chat([{"role": "user", "content": author_prompt(brief, d, au["batch_size"], recent)}],
                                     max_tokens=au["max_tokens"])
            except BaseException:
                stop.set()
                raise
            with lock:
                added = 0
                for item in parse_authored(reply.content):
                    key = seedlib.question_key(item["question"])
                    sh = _shingles(item["question"], 3)
                    similar = any(len(sh & o["_sh"]) / len(sh | o["_sh"]) >= au["near_duplicate"]
                                  for o in by_topic[skill])
                    if key in keys or similar:
                        continue
                    row = {"id": key, "source": "teacher_authored", "generator": skill, "domain": domain,
                           "skill": skill, "difficulty": d, "question": item["question"],
                           "ground_truth": item["answer"], "answer_type": item["answer_type"], "meta": {},
                           "mode": "think"}
                    append_jsonl(p["teacher_authored"], row)
                    keys.add(key)
                    rows.append(row)
                    by_topic[skill].append({**row, "_sh": sh})
                    have[(skill, d)] += 1
                    added += 1
                misses[(skill, d)] += added == 0
                busy.discard((skill, d))
                if state["calls"] % 10 == 0:
                    print(f"author: {len(rows)}/{want} questions after {state['calls']} requests", flush=True)

    run_workers(worker, cfg["teacher"]["concurrency"], stop, "author")
    print(f"author: {len(rows)} questions written by the teacher ({state['calls']} requests this run)", flush=True)


# ---------------------------------------------------------------- solve

def reasoning_cap(seed: dict, quality: dict) -> int:
    """Longest reasoning accepted for this problem, in words. Hard problems get more room;
    select() later keeps such long traces to a minority."""
    hard = int(seed["difficulty"]) >= 4
    return quality["max_reasoning_words_hard"] if hard else quality["max_reasoning_words"]


def judge(seed: dict, reasoning: str, content: str, finish_reason, quality: dict) -> tuple[bool, list[str], str | None]:
    """Decide whether a teacher trace is kept. Returns (keep, reasons_if_not, final_answer)."""
    mode = seed.get("mode", "think")
    q = {**quality, "max_reasoning_words": reasoning_cap(seed, quality)}
    reasons = trace_problems(reasoning, content, finish_reason, q, mode)
    if finish_reason == "reasoning_limit":
        return False, ["reasoning_too_long"], None
    if "truncated" in reasons:
        return False, ["truncated"], None  # everything else wrong with it follows from being cut off
    final = extract_final_answer(content, strict=True)
    if final is None:
        reasons.append("no_final_answer_line")
    elif not check_answer(seed["answer_type"], seed["ground_truth"], final, seed.get("meta")):
        reasons.append("wrong_answer")
    return not reasons, reasons, final


class Scheduler:
    """Hands out the next question.

    Groups (one per generator or topic) are kept roughly level, so the kept set does not
    drift toward whatever the teacher finds easy. A rejected attempt counts for half an
    accepted one, though, so a group the teacher rarely gets through cannot soak up the
    run, and a (group, difficulty) cell with no success after several tries is dropped.
    """

    REJECT_WEIGHT = 0.5
    GIVE_UP_AFTER = 8

    def __init__(self, seeds: list[dict], traces: list[dict], quota: dict[str, int], max_attempts: int):
        self.lock = threading.Lock()
        self.quota, self.max_attempts = quota, max_attempts
        self.attempts = Counter(t["seed_id"] for t in traces)
        accepted_ids = {t["seed_id"] for t in traces if t["ok"]}
        self.acc_src, self.acc_grp, self.rej_grp = Counter(), Counter(), Counter()
        self.fly_src, self.fly_grp = Counter(), Counter()
        self.cell_try, self.cell_acc = Counter(), Counter()
        self.skipped = 0
        self.queues: dict[tuple[str, str], deque] = defaultdict(deque)
        by_id = {s["id"]: s for s in seeds}
        for t in traces:
            s = by_id.get(t["seed_id"])
            if s:
                self.cell_try[self._cell(s)] += 1
                if t["ok"]:
                    self.cell_acc[self._cell(s)] += 1
                else:
                    self.rej_grp[self._group(s)] += 1
        for s in seeds:
            g = self._group(s)
            if s["id"] in accepted_ids:
                self.acc_src[g[0]] += 1
                self.acc_grp[g] += 1
            elif self.attempts[s["id"]] < max_attempts:
                self.queues[g].append(s)

    @staticmethod
    def _group(seed: dict) -> tuple[str, str]:
        return source_of(seed), seed["generator"]

    @staticmethod
    def _cell(seed: dict) -> tuple[str, str, int]:
        return source_of(seed), seed["generator"], int(seed["difficulty"])

    def _hopeless(self, seed: dict) -> bool:
        cell = self._cell(seed)
        return self.cell_acc[cell] == 0 and self.cell_try[cell] >= self.GIVE_UP_AFTER

    def next(self):
        with self.lock:
            while True:
                open_groups = [g for g, q in self.queues.items()
                               if q and self.acc_src[g[0]] + self.fly_src[g[0]] < self.quota[g[0]]]
                if not open_groups:
                    return None
                g = min(open_groups, key=lambda g: (self.acc_grp[g] + self.fly_grp[g]
                                                    + self.REJECT_WEIGHT * self.rej_grp[g], g))
                seed = self.queues[g].popleft()
                if self._hopeless(seed):
                    self.skipped += 1
                    continue
                self.fly_src[g[0]] += 1
                self.fly_grp[g] += 1
                return seed

    def done(self, seed: dict, ok: bool) -> None:
        with self.lock:
            g = self._group(seed)
            self.fly_src[g[0]] -= 1
            self.fly_grp[g] -= 1
            self.attempts[seed["id"]] += 1
            self.cell_try[self._cell(seed)] += 1
            if ok:
                self.acc_src[g[0]] += 1
                self.acc_grp[g] += 1
                self.cell_acc[self._cell(seed)] += 1
            else:
                self.rej_grp[g] += 1
                if self.attempts[seed["id"]] < self.max_attempts:
                    self.queues[g].append(seed)

    def release(self, seed: dict) -> None:
        """Put back a question that was handed out but never sent to the teacher."""
        with self.lock:
            g = self._group(seed)
            self.fly_src[g[0]] -= 1
            self.fly_grp[g] -= 1
            self.queues[g].appendleft(seed)

    def in_flight(self) -> int:
        with self.lock:
            return sum(self.fly_src.values())


def solve(cfg: dict, teacher=None, limit: int | None = None, sources: tuple[str, ...] = SOURCES) -> dict:
    ds, tc, p = cfg["dataset"], cfg["teacher"], paths(cfg)
    teacher = teacher or Teacher(tc)
    seeds = [s for src in sources for s in read_jsonl(p[src])]
    traces = read_jsonl(p["traces"])
    # ask for a little more than the final quota, to cover what select() later drops
    quota = {s: (math.ceil(n * 1.06) if s in sources else 0) for s, n in quotas(cfg).items()}
    sched = Scheduler(seeds, traces, quota, ds["max_attempts"])
    stats = {"calls": 0, "done": 0, "kept": 0, "tokens": 0, "reasons": Counter(), "by_difficulty": {}}
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
                                     direct=direct,
                                     reasoning_limit=None if direct else reasoning_cap(seed, cfg["quality"]))
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
                stats["done"] += 1
                stats["kept"] += ok
                stats["tokens"] += reply.completion_tokens or 0
                stats["reasons"].update(reasons)
                tally = stats["by_difficulty"].setdefault(int(seed["difficulty"]), [0, 0])
                tally[0] += ok
                tally[1] += 1
                n = stats["done"]
                if n % 20 == 0:
                    el = time.time() - t0
                    rate = stats["kept"] / n
                    left = sum(quota.values()) - start_kept - stats["kept"]
                    eta = left / max(rate, 1e-6) * (el / n) / 3600
                    print(f"solve: {n} answers, kept {stats['kept']} ({rate:.0%}), "
                          f"{stats['tokens'] / el:.0f} tok/s, about {eta:.1f} h to go", flush=True)

    run_workers(worker, tc["concurrency"], stop, "solve")
    stats["seconds"] = time.time() - t0
    stats["remaining"] = max(0, sum(quota.values()) - start_kept - stats["kept"])
    by = ", ".join(f"{d}: {k}/{n}" for d, (k, n) in sorted(stats["by_difficulty"].items()))
    print(f"solve: {stats['done']} answers this run, {stats['kept']} kept. Kept by difficulty: {by or 'none'}. "
          f"Rejections: {dict(stats['reasons'].most_common())}", flush=True)
    if sched.skipped:
        print(f"solve: skipped {sched.skipped} problems of kinds the teacher had not once solved within the limits")
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
    long_left = int(ds.get("max_long_share", 1.0) * ds["target_size"])

    def admit(r: dict) -> bool:
        """Long traces are allowed only up to a set share of the dataset."""
        nonlocal long_left
        if r["stats"]["reasoning_words"] <= cfg["quality"]["max_reasoning_words"]:
            return True
        if long_left <= 0:
            return False
        long_left -= 1
        return True

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
                if i < len(groups[k]) and len(picked) < q[src] and admit(groups[k][i]):
                    picked.append(groups[k][i])
            i += 1
        chosen += picked
    # Top up from whatever verified traces are left if one source came in short.
    short = ds["target_size"] - len(chosen)
    if short > 0:
        ids = {r["id"] for r in chosen}
        spare = [r for r in pool.values() if r["id"] not in ids]
        rng.shuffle(spare)
        chosen += [r for r in spare if admit(r)][:short]
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


def stats(cfg: dict) -> None:
    """Keep rates so far, from the trace log. Safe to run while a build is in progress."""
    p = paths(cfg)
    seeds = {s["id"]: s for src in SOURCES for s in read_jsonl(p[src])}
    rows: dict[tuple, Counter] = defaultdict(Counter)
    words: dict[tuple, list[int]] = defaultdict(list)
    gens: dict[str, Counter] = defaultdict(Counter)
    reasons, n = Counter(), 0
    for t in read_jsonl(p["traces"]):
        s = seeds.get(t["seed_id"])
        if not s:
            continue
        n += 1
        k = (source_of(s), int(s["difficulty"]))
        rows[k]["calls"] += 1
        rows[k]["kept"] += t["ok"]
        rows[k]["tokens"] += t["completion_tokens"] or 0
        gens[s["generator"]]["calls"] += 1
        gens[s["generator"]]["kept"] += t["ok"]
        if t["ok"]:
            words[k].append(len(t["reasoning"].split()))
        reasons.update(t["reasons"])
    if not n:
        print("stats: no teacher answers logged yet")
        return
    print(f"{'source':<18}{'difficulty':>10}{'answers':>9}{'kept':>6}{'rate':>7}{'median words':>14}{'tokens/answer':>15}")
    for k in sorted(rows):
        r, w = rows[k], sorted(words[k])
        med = w[len(w) // 2] if w else 0
        print(f"{k[0]:<18}{k[1]:>10}{r['calls']:>9}{r['kept']:>6}{r['kept'] / r['calls']:>7.0%}{med:>14}"
              f"{r['tokens'] / r['calls']:>15.0f}")
    kept = sum(r["kept"] for r in rows.values())
    print(f"total: {kept} kept of {n} ({kept / n:.0%}). Rejections: {dict(reasons.most_common())}")
    worst = sorted((g for g in gens if gens[g]["calls"] >= 4), key=lambda g: gens[g]["kept"] / gens[g]["calls"])[:6]
    if worst:
        print("lowest keep rates: " + ", ".join(f"{g} {gens[g]['kept']}/{gens[g]['calls']}" for g in worst))


def sample(cfg: dict, n: int = 2) -> None:
    """Print a few kept traces in full, for reading. Looking at the data is the best quality check there is."""
    p = paths(cfg)
    seeds = {s["id"]: s for src in SOURCES for s in read_jsonl(p[src])}
    kept = [t for t in read_jsonl(p["traces"]) if t["ok"] and t["seed_id"] in seeds]
    if not kept:
        print("sample: no kept traces yet")
        return
    for t in random.sample(kept, min(n, len(kept))):
        s = seeds[t["seed_id"]]
        print("=" * 78)
        print(f"{s['source']} | {s['skill']} | difficulty {s['difficulty']} | reference answer: {s['ground_truth']}")
        print(f"\nQUESTION\n{s['question']}")
        print(f"\nREASONING ({len(t['reasoning'].split())} words)\n{t['reasoning']}")
        print(f"\nANSWER\n{t['content']}")
    print("=" * 78)


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

def pilot(cfg: dict, teacher=None, calls: int = 60, author_calls: int = 3) -> None:
    """A short trial of the whole dataset path, then a projection for the full build."""
    prepare(cfg)
    teacher = teacher or Teacher(cfg["teacher"])
    q, p, au = quotas(cfg), paths(cfg), cfg["authoring"]
    parts, hours = [], 0.0

    gen = solve(cfg, teacher, limit=calls, sources=("procedural",))
    if gen["done"]:
        rate, per = gen["kept"] / gen["done"], gen["seconds"] / gen["done"]
        h = q["procedural"] * 1.06 / max(rate, 0.01) * per / 3600
        hours += h
        parts.append(f"generated problems about {h:.0f} h")
        print(f"pilot: generated problems: {rate:.0%} kept, {per:.1f} s per answer, "
              f"{gen['tokens'] / gen['seconds']:.0f} tok/s at concurrency {cfg['teacher']['concurrency']}")

    if q["teacher_authored"]:
        before, t0 = len(read_jsonl(p["teacher_authored"])), time.time()
        author(cfg, teacher, limit=author_calls)
        write_secs = time.time() - t0
        written = len(read_jsonl(p["teacher_authored"])) - before
        own = solve(cfg, teacher, limit=max(written, 1), sources=("teacher_authored",)) if written else {"done": 0}
        if written and own["done"]:
            rate, per = own["kept"] / own["done"], own["seconds"] / own["done"]
            needed = q["teacher_authored"] * 1.06 / max(rate, 0.01)
            h = (needed * per + needed / written * write_secs) / 3600
            hours += h
            parts.append(f"teacher-written questions about {h:.0f} h")
            print(f"pilot: teacher-written questions: {written} usable from {author_calls} requests "
                  f"({write_secs / author_calls:.0f} s each); {rate:.0%} kept after solving")
        else:
            print("pilot: the teacher wrote no usable questions in this trial; that part could not be estimated")

    if parts:
        print(f"pilot: projected time for {sum(q.values())} records: roughly {hours:.0f} hours "
              f"({'; '.join(parts)}). This is a small sample, so treat it as a rough guide.")
    top = gen["reasons"].most_common(1) if gen["done"] else []
    if top and top[0][0] == "reasoning_too_short" and top[0][1] > gen["done"] / 2:
        print("pilot: most replies came back with no reasoning. The server is probably not returning the model's "
              "thinking. For Ollama, try `extra_body: {reasoning_effort: medium}` under teacher in config.yaml.")
    if top and top[0][0] == "truncated" and top[0][1] > gen["done"] / 2:
        print("pilot: most replies were cut off. Check the server's context window (see the README, step 1).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "author", "solve", "select", "audit", "all", "pilot", "stats",
                                      "sample"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many teacher calls")
    ap.add_argument("-n", type=int, default=2, help="how many traces `sample` prints")
    a = ap.parse_args(argv)
    cfg = load(a.config)
    if hasattr(sys.stdout, "reconfigure"):  # model text can contain symbols a Windows console code page lacks
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.stage == "prepare":
        prepare(cfg)
    elif a.stage == "author":
        author(cfg, limit=a.limit)
    elif a.stage == "solve":
        solve(cfg, limit=a.limit)
    elif a.stage == "select":
        select(cfg)
    elif a.stage == "stats":
        stats(cfg)
    elif a.stage == "sample":
        sample(cfg, a.n)
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
