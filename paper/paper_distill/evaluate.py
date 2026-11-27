"""Measure a model on held-out problems with exact answers, so Paper can be compared with
its own base model and with its teacher on the same questions.

    python -m paper_distill.evaluate --model base                     # the untouched 4B base
    python -m paper_distill.evaluate                                  # the trained Paper
    python -m paper_distill.evaluate --strategy vote                  # Paper with majority voting
    python -m paper_distill.evaluate --teacher                        # the teacher, via its endpoint
    python -m paper_distill.evaluate --set gsm8k --n 300              # GSM8K test split
    python -m paper_distill.evaluate --retry-unanswered               # re-ask only what got no final answer
    python -m paper_distill.evaluate --summary                        # table of every saved result

The procedural set is generated with a different seed from the training problems and is
checked to share no question with them.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict

from .config import load, read_jsonl
from .infer import HFBackend, HTTPBackend, add_model_args, backend_from_args, parse, solve
from .verify import check_answer


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, mid - half), min(1.0, mid + half)


def load_set(cfg: dict, name: str, n: int | None) -> list[dict]:
    if name == "gsm8k":
        from .seeds.public import gsm8k_test
        return gsm8k_test(n)
    rows = read_jsonl(cfg["data_dir"] / "eval_procedural.jsonl")
    if not rows:
        raise SystemExit("No evaluation set. Run `python -m paper_distill.build prepare` first.")
    return rows if n is None else rows[:n]


def run(backend, system: str, problems: list[dict], inf: dict, strategy: str, batch: int = 8) -> dict:
    gen = dict(temperature=inf["temperature"], top_p=inf["top_p"], max_new_tokens=inf["max_new_tokens"],
               think_budget=inf["think_budget"])
    rows, t0 = [], time.time()
    print(f"evaluate: {len(problems)} problems, {batch} at a time. The first line appears when the first "
          f"{batch} are done, which can take a few minutes.", flush=True)
    for start in range(0, len(problems), batch):
        chunk = problems[start:start + batch]
        if strategy == "single":
            finals = [parse(texts[0])[2] for texts in backend.ask(system, [p["question"] for p in chunk], 1, **gen)]
        else:
            finals = [solve(backend, system, p["question"], inf, "vote")["final_answer"] for p in chunk]
        for p, final in zip(chunk, finals):
            rows.append({"id": p["id"], "domain": p["domain"], "difficulty": p["difficulty"], "final": final,
                         "correct": check_answer(p["answer_type"], p["ground_truth"], final, p.get("meta"))})
        done, el = len(rows), time.time() - t0
        print(f"evaluate: {done}/{len(problems)} done, {sum(r['correct'] for r in rows) / done:.1%} correct so far, "
              f"{el / done:.1f} s each, about {(len(problems) - done) * el / done / 60:.0f} min left", flush=True)

    return {**tabulate(rows), "seconds_per_question": (time.time() - t0) / max(1, len(rows))}


def tabulate(rows: list[dict]) -> dict:
    """Accuracy overall, by difficulty and by domain, from per-problem rows."""
    def score(sub: list[dict]) -> dict:
        k, n = sum(r["correct"] for r in sub), len(sub)
        lo, hi = wilson(k, n)
        return {"n": n, "correct": k, "accuracy": k / n if n else 0.0, "ci95": [round(lo, 4), round(hi, 4)]}

    by_domain, by_diff = defaultdict(list), defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r)
        by_diff[r["difficulty"]].append(r)
    return {"overall": score(rows), "no_answer": sum(r["final"] is None for r in rows),
            "by_difficulty": {str(k): score(v) for k, v in sorted(by_diff.items())},
            "by_domain": {k: score(v) for k, v in sorted(by_domain.items())}, "rows": rows}


def retry_unanswered(backend, system: str, saved: dict, problems: dict[str, dict], inf: dict, rounds: int = 2) -> dict:
    """Ask again, up to `rounds` times, only the problems that produced no final answer in a saved run.

    The result is what that run would have scored had the harness always retried a reply that
    never reached an answer. Problems that were answered, rightly or wrongly, are left untouched.
    """
    gen = dict(temperature=inf["temperature"], top_p=inf["top_p"], max_new_tokens=inf["max_new_tokens"],
               think_budget=inf["think_budget"])
    rows = [dict(r) for r in saved["rows"]]
    for attempt in range(rounds):
        todo = [r for r in rows if r["final"] is None]
        if not todo:
            break
        print(f"evaluate: asking {len(todo)} unanswered problem(s) again (try {attempt + 1} of {rounds})", flush=True)
        replies = backend.ask(system, [problems[r["id"]]["question"] for r in todo], 1, **gen)
        for r, texts in zip(todo, replies):
            p = problems[r["id"]]
            r["final"] = parse(texts[0])[2]
            r["correct"] = check_answer(p["answer_type"], p["ground_truth"], r["final"], p.get("meta"))
    return tabulate(rows)


def summary(cfg: dict) -> None:
    files = sorted((cfg["root"] / "out").glob("eval_*.json"))
    if not files:
        print("No saved results yet.")
        return
    runs = [json.loads(f.read_text()) for f in files]
    print(f"{'model':<28}{'set':<12}{'strategy':<9}{'n':>5}{'accuracy':>10}   {'95% interval':<17}no answer")
    for r in runs:
        o = r["overall"]
        interval = f"{o['ci95'][0]:.1%} to {o['ci95'][1]:.1%}"
        print(f"{r['label']:<28}{r['set']:<12}{r['strategy']:<9}{o['n']:>5}{o['accuracy']:>10.1%}   "
              f"{interval:<17}{r.get('no_answer', 0)}")
    # Same problems, two models: count where one was right and the other wrong. That is a sharper
    # comparison than two separate percentages.
    for i, x in enumerate(runs):
        for y in runs[i + 1:]:
            if x["label"] == y["label"] or (x["set"], x["strategy"]) != (y["set"], y["strategy"]):
                continue
            xr, yr = {r["id"]: r["correct"] for r in x["rows"]}, {r["id"]: r["correct"] for r in y["rows"]}
            both = xr.keys() & yr.keys()
            if not both:
                continue
            only_x = sum(xr[k] and not yr[k] for k in both)
            only_y = sum(yr[k] and not xr[k] for k in both)
            print(f"\n{x['strategy']}, on the same {len(both)} problems: {x['label']} alone was right on {only_x}, "
                  f"{y['label']} alone on {only_y}; they agreed on the other {len(both) - only_x - only_y}.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["procedural", "gsm8k"], default="procedural")
    ap.add_argument("--n", type=int, default=None, help="evaluate only the first n problems")
    ap.add_argument("--strategy", choices=["single", "vote"], default="single")
    ap.add_argument("--teacher", action="store_true", help="evaluate the teacher configured in config.yaml")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--retry-unanswered", action="store_true",
                    help="re-ask only the problems that got no final answer in the saved single run; "
                         "saves the outcome as strategy `retry`")
    ap.add_argument("--batch", type=int, default=8)
    add_model_args(ap)
    a = ap.parse_args(argv)
    cfg = load(a.config)
    if a.summary:
        summary(cfg)
        return 0
    if a.teacher:
        tc = cfg["teacher"]
        backend, system, label = HTTPBackend(tc, tc["system_in_user"]), cfg["instruction"], f"teacher:{tc['model']}"
    elif a.model == "base":
        backend = HFBackend(cfg["student"]["base_model"], None, cfg["train"]["load_4bit"], a.cpu)
        system, label = cfg["system_prompt"], f"base:{cfg['student']['base_model']}"
    else:
        backend, system = backend_from_args(cfg, a)
        label = a.name if a.http else (a.model or "paper")
    out = cfg["root"] / "out"
    out.mkdir(exist_ok=True)
    safe = "".join(c if c.isalnum() else "-" for c in label)
    if a.retry_unanswered:
        first = out / f"eval_{safe}_{a.set}_single.json"
        if not first.exists():
            raise SystemExit(f"No saved single run for {label} on {a.set}. Run the evaluation without "
                             "--retry-unanswered first.")
        problems = {p["id"]: p for p in load_set(cfg, a.set, None)}
        res = retry_unanswered(backend, system, json.loads(first.read_text()), problems, cfg["inference"])
        a.strategy = "retry"
    else:
        res = run(backend, system, load_set(cfg, a.set, a.n), cfg["inference"], a.strategy, a.batch)
    res.update(label=label, set=a.set, strategy=a.strategy)
    path = out / f"eval_{safe}_{a.set}_{a.strategy}.json"
    path.write_text(json.dumps(res, indent=2))
    o = res["overall"]
    print(f"\n{label} on {a.set} ({a.strategy}): {o['correct']}/{o['n']} = {o['accuracy']:.1%} "
          f"(95% interval {o['ci95'][0]:.1%} to {o['ci95'][1]:.1%}), {res['no_answer']} gave no final answer")
    for d, s in res["by_difficulty"].items():
        print(f"  difficulty {d}: {s['accuracy']:.1%} of {s['n']}")
    print(f"saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
