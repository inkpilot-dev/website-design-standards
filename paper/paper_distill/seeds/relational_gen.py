"""Relational, multi-hop, state-tracking and data-interpretation generators."""
from __future__ import annotations

import random

from .base import FEMALE, FMT, MALE, NAMES, Problem, coined, plural


def gen_family(rng: random.Random, d: int) -> Problem:
    males, females = rng.sample(MALE, len(MALE)), rng.sample(FEMALE, len(FEMALE))
    gender: dict[str, str] = {}
    parents: dict[str, tuple[str, str]] = {}  # child -> (father, mother)

    def new(g: str | None = None) -> str:
        g = g or rng.choice("MF")
        name = males.pop() if g == "M" else females.pop()
        gender[name] = g
        return name

    def couple_children(blood: str, lo: int, hi: int) -> list[str]:
        spouse = new("F" if gender[blood] == "M" else "M")
        father, mother = (blood, spouse) if gender[blood] == "M" else (spouse, blood)
        kids = [new() for _ in range(rng.randint(lo, hi))]
        for k in kids:
            parents[k] = (father, mother)
        return kids

    gf, gm = new("M"), new("F")
    kids = [new() for _ in range(2 if d <= 2 else rng.choice([2, 3]))]
    for k in kids:
        parents[k] = (gf, gm)
    blood_parent: dict[str, str] = {}
    grandkids: list[str] = []
    for k in kids:
        for c in couple_children(k, 1, 2 if d <= 1 else 3):
            blood_parent[c] = k
            grandkids.append(c)
    greats: list[str] = []
    if d >= 4:
        through = rng.choice(grandkids)
        for c in couple_children(through, 1, 2):
            blood_parent[c] = through
            greats.append(c)
    facts = []
    for c, (f, m) in parents.items():
        if rng.random() < 0.5:
            facts += [f"{f} is the father of {c}.", f"{m} is the mother of {c}."]
        else:
            facts.append(f"{c}'s father is {f} and {c}'s mother is {m}.")
    if d >= 3:  # an unrelated family as a distractor
        of, om = new("M"), new("F")
        for _ in range(rng.randint(1, 2)):
            c = new()
            facts += [f"{of} is the father of {c}.", f"{om} is the mother of {c}."]
    rng.shuffle(facts)
    everyone = list(gender)
    role = lambda person: "father" if gender[person] == "M" else "mother"  # noqa: E731

    def name_q(q: str, ans: str) -> Problem:
        return Problem(f"{' '.join(facts)} {q} {FMT['name']}", ans, "text", "relational", "family_relations", d,
                       {"distractors": [p for p in everyone if p != ans]})

    def int_q(q: str, ans: int) -> Problem:
        return Problem(f"{' '.join(facts)} {q} {FMT['int']}", str(ans), "int", "relational", "family_relations", d)

    def set_q(q: str, ans: list[str]) -> Problem:
        return Problem(f"{' '.join(facts)} {q} {FMT['set']}", ", ".join(sorted(ans)), "set", "relational",
                       "family_relations", d)

    v = rng.choice({1: ["grand"], 2: ["grand", "count_gc"], 3: ["cousins_n", "cousins"],
                    4: ["chain", "count_great"], 5: ["chain", "descendants", "others"]}[d])
    if v == "grand":
        c = rng.choice(grandkids)
        side = "paternal" if gender[blood_parent[c]] == "M" else "maternal"
        which = rng.choice(["grandfather", "grandmother"])
        return name_q(f"Who is the {side} {which} of {c}?", gf if which == "grandfather" else gm)
    if v == "count_gc":
        return int_q(f"How many grandchildren does {rng.choice([gf, gm])} have?", len(grandkids))
    if v in ("cousins_n", "cousins"):
        c = rng.choice(grandkids)
        cousins = [g for g in grandkids if blood_parent[g] != blood_parent[c]]
        if v == "cousins_n":
            return int_q(f"How many first cousins does {c} have among the people mentioned?", len(cousins))
        return set_q(f"Who are the first cousins of {c} among the people mentioned?", cousins)
    if v == "chain":
        g = rng.choice(greats)
        p1 = blood_parent[g]
        p2 = blood_parent[p1]
        top = rng.choice([gf, gm])
        return name_q(f"Who is the {role(top)} of the {role(p2)} of the {role(p1)} of {g}?", top)
    if v == "count_great":
        return int_q(f"How many great-grandchildren does {rng.choice([gf, gm])} have?", len(greats))
    if v == "descendants":
        return int_q(f"How many of the people mentioned are descendants of {gm} (children, grandchildren and "
                     "great-grandchildren)?", len(kids) + len(grandkids) + len(greats))
    k = rng.choice(kids)
    return set_q(f"Which grandchildren of {gf} are not children of {k}?",
                 [g for g in grandkids if blood_parent[g] != k])


_COMPARE = [("taller", "shorter", "tallest", "shortest"), ("older", "younger", "oldest", "youngest"),
            ("faster", "slower", "fastest", "slowest"), ("heavier", "lighter", "heaviest", "lightest")]
_MEASURE = [("cm", "tall", "taller", "shorter", (150, 195), "tallest", "shortest"),
            ("years", "old", "older", "younger", (20, 70), "oldest", "youngest"),
            ("kg", "in weight", "heavier", "lighter", (45, 110), "heaviest", "lightest")]


def gen_comparison(rng: random.Random, d: int) -> Problem:
    n = min(3 + d, 7)
    people = rng.sample(NAMES, n)
    if rng.random() < 0.5:
        more, less, most, least = rng.choice(_COMPARE)
        facts = []
        for a, b in zip(people, people[1:]):  # people is ordered from most to least
            facts.append(f"{a} is {more} than {b}." if rng.random() < 0.5 else f"{b} is {less} than {a}.")
        rng.shuffle(facts)
        k = rng.randrange(n)
        label = most if k == 0 else least if k == n - 1 else f"{['', 'second', 'third', 'fourth', 'fifth', 'sixth'][k]} {most}"
        ans = people[k]
        return Problem(f"{' '.join(facts)} Who is the {label}? {FMT['name']}", ans, "text", "relational",
                       "transitive_comparison", d, {"distractors": [p for p in people if p != ans]})
    unit, adj, more, less, (lo, hi), most, least = rng.choice(_MEASURE)
    while True:
        vals = {people[0]: rng.randint(lo, hi)}
        facts = [f"{people[0]} is {vals[people[0]]} {unit} {adj}."]
        for p in people[1:]:
            ref = rng.choice(list(vals))
            diff = rng.randint(1, 15)
            sign = rng.choice([1, -1])
            vals[p] = vals[ref] + sign * diff
            facts.append(f"{p} is {diff} {unit} {more if sign > 0 else less} than {ref}.")
        srt = sorted(vals.values())
        if srt[0] > 0 and srt[-1] != srt[-2]:
            break
    head, rest = facts[0], facts[1:]
    rng.shuffle(rest)
    facts = rest[:len(rest) // 2] + [head] + rest[len(rest) // 2:]
    v = rng.choice(["value", "value", "range"] if d >= 3 else ["value"])
    if v == "value":
        target = people[-1]
        q = f"How many {unit} {adj} is {target}?" if adj != "in weight" else f"What is {target}'s weight in {unit}?"
        ans = vals[target]
    else:
        q = f"What is the difference in {unit} between the {most} and the {least} person?"
        ans = srt[-1] - srt[0]
    return Problem(f"{' '.join(facts)} {q} {FMT['int']}", str(ans), "int", "relational",
                   "numeric_comparison", d)


_DIRS = ["north", "east", "south", "west"]


def gen_navigation(rng: random.Random, d: int) -> Problem:
    name = rng.choice(NAMES)
    while True:
        heading, x, y, steps, turns = 0, 0, 0, [], 0
        for i in range(3 + 2 * d):
            if i % 2 == 0 or rng.random() < 0.25:
                k = rng.randint(1, 9)
                dx, dy = [(0, 1), (1, 0), (0, -1), (-1, 0)][heading]
                x, y = x + dx * k, y + dy * k
                steps.append(f"walks {plural(k, 'block')} forward")
            else:
                t = rng.choice(["left", "right", "around"])
                heading = (heading + {"left": 3, "right": 1, "around": 2}[t]) % 4
                steps.append("turns around" if t == "around" else f"turns 90 degrees to the {t}")
                turns += 1
        if turns >= 1:
            break
    body = (f"{name} starts at a street corner facing north and then does the following in order: "
            + "; ".join(f"({i + 1}) {s}" for i, s in enumerate(steps)) + ".")
    v = rng.choice(["dist", "dist", "facing", "east"] if d >= 2 else ["dist", "facing"])
    if v == "facing":
        ans = _DIRS[heading]
        return Problem(f"{body} Which compass direction is {name} facing at the end? Answer with one word.", ans,
                       "text", "spatial_temporal", "navigation", d, {"distractors": [x_ for x_ in _DIRS if x_ != ans]})
    if v == "dist":
        q = (f"How many blocks is {name} from the starting corner at the end, counting the east-west blocks "
             "plus the north-south blocks?")
        ans = abs(x) + abs(y)
    else:
        q = f"At the end, how many blocks east or west of the starting corner is {name}, ignoring north-south distance?"
        ans = abs(x)
    return Problem(f"{body} {q} {FMT['int']}", str(ans), "int", "spatial_temporal", "navigation", d)


_OBJECTS = ["red ball", "blue notebook", "green umbrella", "silver key", "yellow hat", "white mug"]


def gen_tracking(rng: random.Random, d: int) -> Problem:
    n = 3 if d <= 2 else 4 if d <= 4 else 5
    people = rng.sample(NAMES, n)
    objs = rng.sample(_OBJECTS, n)
    holds = dict(zip(people, objs))
    start = "; ".join(f"{p} has the {o}" for p, o in holds.items())
    events = []
    for _ in range(2 + 2 * d):
        if d >= 3 and rng.random() < 0.3:
            a, b, c = rng.sample(people, 3)
            holds[b], holds[c], holds[a] = holds[a], holds[b], holds[c]
            events.append(f"{a} passes their item to {b}, {b} passes theirs to {c}, and {c} passes theirs to {a}, "
                          "all at the same moment")
        else:
            a, b = rng.sample(people, 2)
            holds[a], holds[b] = holds[b], holds[a]
            events.append(f"{a} and {b} swap items")
    body = f"At the start: {start}. Then, in order: " + "; ".join(f"({i + 1}) {e}" for i, e in enumerate(events)) + "."
    if rng.random() < 0.5:
        o = rng.choice(objs)
        ans = next(p for p, h in holds.items() if h == o)
        return Problem(f"{body} Who has the {o} at the end? {FMT['name']}", ans, "text", "state_tracking",
                       "object_tracking", d, {"distractors": [p for p in people if p != ans]})
    p = rng.choice(people)
    ans = holds[p]
    return Problem(f"{body} Which item does {p} have at the end? Answer with the item only.", ans, "text",
                   "state_tracking", "object_tracking", d, {"distractors": [o for o in objs if o != ans]})


def gen_multihop(rng: random.Random, d: int) -> Problem:
    hops = min(d + 1, 4)
    k = 3 + (d >= 3) + (d >= 5)
    cap = lambda: coined(rng, 2).capitalize()  # noqa: E731
    used: set[str] = set()

    def fresh(make) -> list[str]:
        out = []
        while len(out) < k:
            w = make()
            if w not in used and all(w not in u and u not in w for u in used):
                used.add(w)
                out.append(w)
        return out

    people = fresh(lambda: f"{cap()} {cap()}")
    firms = fresh(lambda: f"{cap()} {rng.choice(['Works', 'Labs', 'Group', 'Foods', 'Motors'])}")
    cities = fresh(cap)
    regions = fresh(lambda: cap() + "ia")
    governors = fresh(lambda: f"{cap()} {cap()}")
    works = {p: rng.choice(firms) for p in people}
    hq = {f: rng.choice(cities) for f in firms}
    region = {c: rng.choice(regions) for c in cities}
    gov = dict(zip(regions, governors))
    facts = ([f"{p} works for {f}." for p, f in works.items()] +
             [f"{f} has its headquarters in {c}." for f, c in hq.items()] +
             [f"{c} is in the province of {r}." for c, r in region.items()] +
             [f"The governor of {r} is {g}." for r, g in gov.items()])
    rng.shuffle(facts)
    p = rng.choice(people)
    if hops == 2:
        q, ans, pool = f"In which city is the headquarters of the company that {p} works for?", hq[works[p]], cities
    elif hops == 3:
        q = f"In which province is the city where {p}'s employer has its headquarters?"
        ans, pool = region[hq[works[p]]], regions
    else:
        q = f"Who is the governor of the province containing the city where {p}'s employer has its headquarters?"
        ans, pool = gov[region[hq[works[p]]]], governors
    return Problem(f"{' '.join(facts)} {q} Answer with the name only.", ans, "text", "multi_hop",
                   "fact_composition", d, {"distractors": [x for x in pool if x != ans]})


_VENN = [("students", ["the chess club", "the drama club", "the robotics club"], "are in"),
         ("customers", ["tea", "coffee", "juice"], "bought"),
         ("employees", ["Spanish", "French", "German"], "speak")]


def gen_venn(rng: random.Random, d: int) -> Problem:
    who, sets, verb = rng.choice(_VENN)
    if d <= 2:
        a, b = sets[:2]
        oa, ob, both, none = rng.randint(5, 30), rng.randint(5, 30), rng.randint(2, 20), rng.randint(0, 20)
        total = oa + ob + both + none
        v = rng.choice(["none", "only", "one", "both"])
        if v == "both":
            facts = (f"In a group of {total} {who}, {oa + both} {verb} {a}, {ob + both} {verb} {b}, and "
                     f"{none} {verb} neither.")
            q, ans = f"How many {verb} both?", both
        else:
            facts = (f"In a group of {total} {who}, {oa + both} {verb} {a}, {ob + both} {verb} {b}, and "
                     f"{both} {verb} both.")
            q, ans = {"none": (f"How many {verb} neither?", none), "only": (f"How many {verb} {a} but not {b}?", oa),
                      "one": (f"How many {verb} exactly one of the two?", oa + ob)}[v]
    else:
        a, b, c = sets
        oa, ob, oc = (rng.randint(3, 20) for _ in range(3))
        ab, ac, bc = (rng.randint(1, 10) for _ in range(3))
        abc, none = rng.randint(1, 8), rng.randint(0, 15)
        total = oa + ob + oc + ab + ac + bc + abc + none
        facts = (f"In a group of {total} {who}: {oa + ab + ac + abc} {verb} {a}, {ob + ab + bc + abc} {verb} {b}, "
                 f"and {oc + ac + bc + abc} {verb} {c}. {ab + abc} {verb} both {a} and {b}, {ac + abc} {verb} both "
                 f"{a} and {c}, and {bc + abc} {verb} both {b} and {c}; each of these three counts includes the "
                 f"{abc} who {verb} all three.")
        q, ans = rng.choice([
            (f"How many {verb} none of the three?", none),
            (f"How many {verb} exactly one of the three?", oa + ob + oc),
            (f"How many {verb} exactly two of the three?", ab + ac + bc),
            (f"How many {verb} {a} only?", oa),
            (f"How many {verb} at least two of the three?", ab + ac + bc + abc),
        ])
    return Problem(f"{facts} {q} {FMT['int']}", str(ans), "int", "combinatorics_probability", "inclusion_exclusion", d)


_DEPTS = ["Sales", "Design", "Support", "Finance"]


def gen_table(rng: random.Random, d: int) -> Problem:
    while True:
        n = 5 + d
        people = rng.sample(NAMES, n)
        depts = rng.sample(_DEPTS, 3)
        rows = [(p, rng.choice(depts), rng.randrange(40, 131, 5), rng.randint(1, 20)) for p in people]
        by = {dep: [r for r in rows if r[1] == dep] for dep in depts}
        if any(not v for v in by.values()):
            continue
        totals = {dep: sum(r[2] for r in v) for dep, v in by.items()}
        sizes = {dep: len(v) for dep, v in by.items()}
        table = "Name | Department | Salary ($ thousand) | Years at company\n" + "\n".join(
            f"{p} | {dep} | {s} | {y}" for p, dep, s, y in rows)
        intro = f"Here is a table of employees:\n{table}\n"
        if d == 1:
            dep = rng.choice(depts)
            return _int(intro + f"What is the total salary, in thousands of dollars, of the {dep} department?",
                        totals[dep], d)
        if d == 2:
            best = max(totals, key=totals.get)
            if sorted(totals.values())[-1] == sorted(totals.values())[-2]:
                continue
            return Problem(intro + "Which department has the highest total salary? Answer with the department name only.",
                           best, "text", "data_interpretation", "table_reasoning", d,
                           {"distractors": [x for x in depts if x != best]})
        if d == 3:
            y = rng.randint(4, 12)
            pool = [r for r in rows if r[3] > y]
            if len(pool) < 2:
                continue
            low = min(pool, key=lambda r: r[2])
            if sum(1 for r in pool if r[2] == low[2]) > 1:
                continue
            return Problem(intro + f"Among employees with more than {y} years at the company, who has the lowest "
                           f"salary? {FMT['name']}", low[0], "text", "data_interpretation", "table_reasoning", d,
                           {"distractors": [p for p in people if p != low[0]]})
        if d == 4:
            avg_total = sum(r[2] for r in rows)
            ans = sum(1 for r in rows if r[2] * n > avg_total)
            return _int(intro + "How many employees earn more than the average salary of everyone in the table?",
                        ans, d)
        small = min(sizes, key=sizes.get)
        if sorted(sizes.values())[0] == sorted(sizes.values())[1]:
            continue
        rows10 = [(p, dep, s * 10, y) for p, dep, s, y in rows]  # work in $ hundreds to keep 10% exact
        new_total = sum(s * 11 // 10 if dep == small else s for _, dep, s, _ in rows10)
        return _int(intro + "Every employee in the department with the fewest people gets a 10% raise. What is the "
                    "new total salary of the whole table, in hundreds of dollars?", new_total, d)


def _int(q: str, ans: int, d: int) -> Problem:
    return Problem(f"{q} {FMT['int']}", str(ans), "int", "data_interpretation", "table_reasoning", d)


GENERATORS = [gen_family, gen_comparison, gen_navigation, gen_tracking, gen_multihop, gen_venn, gen_table]
