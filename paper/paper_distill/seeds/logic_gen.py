"""Deduction puzzles. Every puzzle is checked by brute force to have exactly one solution."""
from __future__ import annotations

import itertools
import random

from .base import FMT, NAMES, Problem, coined_set


def _unique_clues(rng, universe, pool, max_clues):
    """Greedily pick clues until one candidate is left, then drop any that became redundant.

    pool: (text, predicate) pairs that are all true of the hidden solution.
    Returns the clue list, or None if the pool cannot pin the solution down.
    """
    clues, cands = [], list(universe)
    for text, pred in pool:
        kept = [c for c in cands if pred(c)]
        if len(kept) < len(cands):
            clues.append((text, pred))
            cands = kept
        if len(cands) == 1:
            break
    if len(cands) != 1:
        return None

    def unique(cs) -> bool:
        seen = 0
        for c in universe:
            if all(p(c) for _, p in cs):
                seen += 1
                if seen > 1:
                    return False
        return seen == 1

    i = 0
    while i < len(clues):
        rest = clues[:i] + clues[i + 1:]
        if unique(rest):
            clues = rest
        else:
            i += 1
    return clues if len(clues) <= max_clues else None


def gen_ordering(rng: random.Random, d: int) -> Problem:
    n = 4 if d <= 2 else 5 if d == 3 else 6
    while True:
        people = rng.sample(NAMES, n)
        truth = tuple(people)
        universe = list(itertools.permutations(sorted(people)))
        weak, strong = [], []
        for x, y in itertools.permutations(people, 2):
            dx = truth.index(y) - truth.index(x)
            if dx == 1:
                strong.append((f"{x} is immediately to the left of {y}.",
                               lambda c, x=x, y=y: c.index(y) - c.index(x) == 1))
            if dx >= 2:
                weak.append((f"{x} is somewhere to the left of {y}.",
                             lambda c, x=x, y=y: c.index(x) < c.index(y)))
            if x < y and abs(dx) >= 2:
                g = abs(dx) - 1
                between = "is exactly 1 person" if g == 1 else f"are exactly {g} people"
                weak.append((f"There {between} between {x} and {y}.",
                             lambda c, x=x, y=y, g=g: abs(c.index(x) - c.index(y)) == g + 1))
                weak.append((f"{x} is not next to {y}.",
                             lambda c, x=x, y=y: abs(c.index(x) - c.index(y)) != 1))
        for x in people:
            i = truth.index(x)
            if i in (0, n - 1):
                strong.append((f"{x} is at one of the two ends.", lambda c, x=x: c.index(x) in (0, n - 1)))
            else:
                weak.append((f"{x} is not at either end.", lambda c, x=x: c.index(x) not in (0, n - 1)))
            if d <= 2:
                strong.append((f"{x} is in position {i + 1}.", lambda c, x=x, i=i: c.index(x) == i))
        rng.shuffle(weak)
        rng.shuffle(strong)
        if d <= 2:
            pool = weak + strong
            rng.shuffle(pool)
        else:
            pool = weak + strong  # indirect clues first makes the puzzle take more steps
        clues = _unique_clues(rng, universe, pool, max_clues=4 + d)
        if clues:
            break
    texts = [t for t, _ in clues]
    rng.shuffle(texts)
    intro = (f"{n} people ({', '.join(sorted(people))}) stand in a single row. Positions are numbered "
             f"1 to {n} from left to right.")
    body = intro + " " + " ".join(texts)
    if rng.random() < 0.6:
        k = rng.randrange(n)
        ans = truth[k]
        return Problem(f"{body} Who is in position {k + 1}? {FMT['name']}", ans, "text", "logic_deduction",
                       "ordering_puzzle", d, {"distractors": [p for p in people if p != ans]})
    return Problem(f"{body} What is the full order from left to right? {FMT['list']}", ", ".join(truth),
                   "list", "logic_deduction", "ordering_puzzle", d)


_CATS = {
    "pet": (["cat", "dog", "parrot", "rabbit"], "owns the {v}", "own the {v}", "Who owns the {v}?",
            "Which pet does {p} own?"),
    "drink": (["tea", "coffee", "juice", "milk"], "drinks {v}", "drink {v}", "Who drinks {v}?",
              "What does {p} drink?"),
    "hobby": (["chess", "painting", "cycling", "baking"], "enjoys {v}", "enjoy {v}", "Who enjoys {v}?",
              "Which hobby does {p} enjoy?"),
}


def gen_assignment(rng: random.Random, d: int) -> Problem:
    n = 3 if d <= 2 else 4
    cats = rng.sample(list(_CATS), 2 if d <= 3 else 3)
    while True:
        people = rng.sample(NAMES, n)
        values = {c: rng.sample(_CATS[c][0], n) for c in cats}
        truth = tuple(tuple(rng.sample(values[c], n)) for c in cats)  # truth[ci][person] = value
        universe = list(itertools.product(*[list(itertools.permutations(values[c])) for c in cats]))
        direct, indirect = [], []
        for ci, c in enumerate(cats):
            pos, neg = _CATS[c][1], _CATS[c][2]
            for pi, p in enumerate(people):
                for v in values[c]:
                    if truth[ci][pi] == v:
                        direct.append((f"{p} {pos.format(v=v)}.", lambda s, ci=ci, pi=pi, v=v: s[ci][pi] == v))
                    else:
                        indirect.append((f"{p} does not {neg.format(v=v)}.",
                                         lambda s, ci=ci, pi=pi, v=v: s[ci][pi] != v))
        for (c1i, c1), (c2i, c2) in itertools.combinations(list(enumerate(cats)), 2):
            for v1 in values[c1]:
                for v2 in values[c2]:
                    same = truth[c1i].index(v1) == truth[c2i].index(v2)
                    ref = f"The person who {_CATS[c1][1].format(v=v1)}"
                    if same:
                        indirect.append((f"{ref} {_CATS[c2][1].format(v=v2)}.",
                                         lambda s, a=c1i, b=c2i, v1=v1, v2=v2: s[a].index(v1) == s[b].index(v2)))
                    else:
                        indirect.append((f"{ref} does not {_CATS[c2][2].format(v=v2)}.",
                                         lambda s, a=c1i, b=c2i, v1=v1, v2=v2: s[a].index(v1) != s[b].index(v2)))
        rng.shuffle(direct)
        rng.shuffle(indirect)
        if d <= 2:
            pool = direct[:2] + indirect
            rng.shuffle(pool)
            pool += direct[2:]
        else:
            pool = indirect + direct
        clues = _unique_clues(rng, universe, pool, max_clues=5 + 2 * d)
        if clues:
            break
    texts = [t for t, _ in clues]
    rng.shuffle(texts)
    plurals = {"pet": "pets", "drink": "drinks", "hobby": "hobbies"}
    lists = "; ".join(f"the {plurals[c]} are {', '.join(sorted(values[c]))}" for c in cats)
    intro = (f"{', '.join(people[:-1])} and {people[-1]} each have exactly one {' and one '.join(cats)}, "
             f"and no two of them share the same one ({lists}).")
    ci = rng.randrange(len(cats))
    c = cats[ci]
    pi = rng.randrange(n)
    if rng.random() < 0.5:
        q, ans, distract = _CATS[c][3].format(v=truth[ci][pi]), people[pi], [p for p in people if p != people[pi]]
        fmt = FMT["name"]
    else:
        q, ans = _CATS[c][4].format(p=people[pi]), truth[ci][pi]
        distract, fmt = [v for v in values[c] if v != ans], "Answer with one word."
    return Problem(f"{intro} {' '.join(texts)} {q} {fmt}", ans, "text", "logic_deduction",
                   "assignment_puzzle", d, {"distractors": distract})


def gen_syllogism(rng: random.Random, d: int) -> Problem:
    n = min(3 + d, 7)
    while True:
        terms = [w + ("es" if w.endswith("s") else "s") for w in coined_set(rng, n)]
        alls = {(i, i + 1) for i in range(n - 1) if rng.random() < 0.75}
        for _ in range(rng.randint(0, 2)):
            i, j = sorted(rng.sample(range(n), 2))
            alls.add((i, j))
        nos = set()
        for _ in range(rng.randint(1, 2)):
            i, j = sorted(rng.sample(range(n), 2))
            nos.add((i, j))
        if len(alls) < 2:
            continue
        # A "type" is one possible membership pattern for an individual. A universal
        # conclusion follows exactly when no allowed type contradicts it.
        types = [t for t in range(1, 2 ** n)
                 if all(not (t >> i & 1) or (t >> j & 1) for i, j in alls)
                 and all(not ((t >> i & 1) and (t >> j & 1)) for i, j in nos)]
        if any(not any(t >> i & 1 for t in types) for i in range(n)):
            continue  # a category would be forced empty; avoid vacuous reasoning
        follows, not_follows = [], []
        for x, y in itertools.permutations(range(n), 2):
            if (x, y) not in alls:
                ok = all(not (t >> x & 1) or (t >> y & 1) for t in types)
                (follows if ok else not_follows).append(("all", x, y))
            if x < y and (x, y) not in nos:
                ok = all(not ((t >> x & 1) and (t >> y & 1)) for t in types)
                (follows if ok else not_follows).append(("no", x, y))
        if follows and not_follows:
            break
    want = rng.random() < 0.5
    kind, x, y = rng.choice(follows if want else not_follows)
    stmts = [f"All {terms[i]} are {terms[j]}." for i, j in alls] + [f"No {terms[i]} are {terms[j]}." for i, j in nos]
    rng.shuffle(stmts)
    concl = f"all {terms[x]} are {terms[y]}" if kind == "all" else f"no {terms[x]} are {terms[y]}"
    q = ("Here are some statements about made-up categories:\n" + "\n".join(f"- {s}" for s in stmts) +
         f"\nBased only on these statements, does it necessarily follow that {concl}? {FMT['yesno']}")
    return Problem(q, "Yes" if want else "No", "yesno", "logic_deduction", "syllogisms", d)


def gen_truth_tellers(rng: random.Random, d: int) -> Problem:
    n = 3 if d <= 2 else 4 if d <= 4 else 5
    kinds = ["liar", "truth", "same", "diff", "count"] + (["atleast", "both"] if d >= 3 else [])
    while True:
        people = rng.sample(NAMES, n)
        stmts = []
        for i in range(n):
            others = [j for j in range(n) if j != i]
            k = rng.choice(kinds)
            j, m = rng.sample(others, 2)
            if k == "liar":
                stmts.append((f"{people[j]} is a liar.", lambda t, j=j: not t[j]))
            elif k == "truth":
                stmts.append((f"{people[j]} is a truth-teller.", lambda t, j=j: t[j]))
            elif k == "same":
                stmts.append((f"{people[j]} and {people[m]} are the same type.", lambda t, j=j, m=m: t[j] == t[m]))
            elif k == "diff":
                stmts.append((f"{people[j]} and {people[m]} are different types.", lambda t, j=j, m=m: t[j] != t[m]))
            elif k == "count":
                c = rng.randint(1, n - 1)
                stmts.append((f"Exactly {c} of the {n} of us {'is a truth-teller' if c == 1 else 'are truth-tellers'}.",
                              lambda t, c=c: sum(t) == c))
            elif k == "atleast":
                stmts.append((f"At least one of {people[j]} and {people[m]} is a liar.",
                              lambda t, j=j, m=m: not (t[j] and t[m])))
            else:
                stmts.append((f"{people[j]} and {people[m]} are both truth-tellers.", lambda t, j=j, m=m: t[j] and t[m]))
        sols = [t for t in itertools.product([True, False], repeat=n)
                if all(bool(f(t)) == t[i] for i, (_, f) in enumerate(stmts))]
        if len(sols) == 1 and any(sols[0]) and not all(sols[0]):
            break
    sol = sols[0]
    said = " ".join(f'{people[i]} says: "{s}"' for i, (s, _) in enumerate(stmts))
    intro = (f"Each of {n} people ({', '.join(people)}) is either a truth-teller, whose statements are all true, "
             "or a liar, whose statements are all false.")
    v = rng.choice(["count", "is", "which"])
    if v == "count":
        return Problem(f"{intro} {said} How many of them are truth-tellers? {FMT['int']}", str(sum(sol)), "int",
                       "logic_deduction", "truth_tellers_and_liars", d)
    if v == "is":
        i = rng.randrange(n)
        return Problem(f"{intro} {said} Is {people[i]} a truth-teller? {FMT['yesno']}", "Yes" if sol[i] else "No",
                       "yesno", "logic_deduction", "truth_tellers_and_liars", d)
    tellers = sorted(p for p, t in zip(people, sol) if t)
    return Problem(f"{intro} {said} Which of them are truth-tellers? {FMT['set']}", ", ".join(tellers), "set",
                   "logic_deduction", "truth_tellers_and_liars", d)


GENERATORS = [gen_ordering, gen_assignment, gen_syllogism, gen_truth_tellers]
