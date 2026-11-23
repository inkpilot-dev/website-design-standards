"""Arithmetic, algebra, number theory, counting and geometry generators."""
from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction

from .base import FMT, Problem, frac_str, ordinal, plural

_SKINS = [("A bakery", "muffins", "trays"), ("A bookshop", "notebooks", "boxes"),
          ("A warehouse", "lamps", "crates"), ("A farm stand", "eggs", "cartons"),
          ("A school store", "pencils", "packs"), ("A nursery", "seedlings", "flats")]
_FRACS = [(1, 2, "half"), (1, 3, "one third"), (1, 4, "one quarter"), (2, 3, "two thirds"),
          (3, 4, "three quarters"), (1, 5, "one fifth"), (2, 5, "two fifths")]


def gen_inventory(rng: random.Random, d: int) -> Problem:
    owner, items, box = rng.choice(_SKINS)
    total = rng.randint(60, 360)
    lines = [f"{owner} begins the week with {total} {items}."]
    for _ in range(d + 2):
        fr = [f for f in _FRACS if total > 0 and total % f[1] == 0]
        kind = rng.choice(["add", "add", "sell", "return"] + (["frac", "frac"] if fr else []))
        if kind == "sell" and total < 12:
            kind = "add"
        if kind == "add":
            a, b = rng.randint(2, 9), rng.randint(6, 24)
            total += a * b
            lines.append(f"It receives {a} {box} holding {b} {items} each.")
        elif kind == "sell":
            c = rng.randint(5, total // 2)
            total -= c
            lines.append(f"It sells {c} {items}.")
        elif kind == "return":
            r = rng.randint(2, 15)
            total += r
            lines.append(f"Customers bring back {r} {items}.")
        else:
            a, b, w = rng.choice(fr)
            total -= total * a // b
            lines.append(f"It then sells {w} of the {items} it has at that moment.")
    if rng.random() < 0.5:
        q, ans = f"How many {items} does it have at the end?", total
    else:
        m = rng.choice([4, 5, 6, 8, 10, 12])
        q = (f"At the end, the remaining {items} are packed into {box} of {m}. "
             f"How many full {box} can be filled?")
        ans = total // m
    return Problem(" ".join(lines) + f" {q} {FMT['int']}", str(ans), "int",
                   "arithmetic", "multi_step_word_problem", d)


def gen_rates(rng: random.Random, d: int) -> Problem:
    v = rng.choice(["pipes", "trains", "avg"])
    if v == "pipes":
        a, b = rng.sample(range(2, 13), 2)
        if d <= 2:
            t = 1 / (Fraction(1, a) + Fraction(1, b))
            q = (f"Pipe A can fill a tank in {a} hours and pipe B can fill it in {b} hours. "
                 "Starting with an empty tank, how many hours does it take to fill it with both pipes open?")
        elif d == 3:
            while True:
                c = rng.randint(3, 20)
                rate = Fraction(1, a) + Fraction(1, b) - Fraction(1, c)
                if rate > 0:
                    break
            t = 1 / rate
            q = (f"Pipe A fills a tank in {a} hours, pipe B fills it in {b} hours, and an open drain "
                 f"empties a full tank in {c} hours. Starting empty, with both pipes and the drain open, "
                 "how many hours pass until the tank is full?")
        else:
            s = rng.randint(1, a - 1)
            t = s + (1 - Fraction(s, a)) / (Fraction(1, a) + Fraction(1, b))
            q = (f"Pipe A fills a tank in {a} hours and pipe B fills it in {b} hours. Pipe A runs alone "
                 f"for {plural(s, 'hour')}, then pipe B is opened as well. How many hours in total, "
                 "counted from the start, does it take to fill the tank?")
        ans, unit = t, "in hours"
    elif v == "trains":
        while True:
            dist = rng.randrange(120, 601, 20)
            v1, v2 = rng.randrange(40, 111, 5), rng.randrange(40, 111, 5)
            h = 0 if d <= 2 else rng.choice([1, 2])
            if v1 * h < dist:
                break
        t = Fraction(dist + v2 * h, v1 + v2)
        start = ("Both trains leave at the same time." if h == 0 else
                 f"The train from B leaves {plural(h, 'hour')} after the train from A.")
        base = (f"Stations A and B are {dist} km apart. A train leaves A toward B at {v1} km/h and "
                f"another leaves B toward A at {v2} km/h. {start}")
        if d >= 5:
            q, ans, unit = f"{base} How far from station A do the trains meet?", v1 * t, "in km"
        else:
            q = f"{base} How many hours after the train from A departs do the two trains meet?"
            ans, unit = t, "in hours"
    else:
        legs = [(rng.randrange(10, 91, 5), rng.randrange(10, 61, 5)) for _ in range(2 if d <= 2 else 3)]
        total_d = sum(x for x, _ in legs)
        total_t = sum(Fraction(x, s) for x, s in legs)
        parts = ", then ".join(f"{x} km at {s} km/h" for x, s in legs)
        q = f"A cyclist rides {parts}. What is the cyclist's average speed over the whole trip?"
        ans, unit = total_d / total_t, "in km/h"
    return Problem(f"{q} Give the answer {unit} as a simplified fraction, or as a whole number if it is one.",
                   frac_str(ans), "fraction", "arithmetic", "rates", d)


def gen_percent(rng: random.Random, d: int) -> Problem:
    item = rng.choice(["jacket", "bicycle", "lamp", "tablet", "sofa", "watch"])
    n = 1 if d == 1 else 2 if d in (2, 4) else 3
    while True:
        p = rng.randrange(20, 401, 5)
        kinds = rng.sample(["up", "down", "tax"], n)
        kinds.sort(key=lambda k: k == "tax")  # tax is applied last
        factor, parts = Fraction(1), []
        for k in kinds:
            if k == "up":
                a = rng.choice([10, 20, 25, 30, 40, 50])
                factor *= Fraction(100 + a, 100)
                parts.append(f"raised by {a}%")
            elif k == "down":
                b = rng.choice([10, 15, 20, 25, 30, 40, 50])
                factor *= Fraction(100 - b, 100)
                parts.append(f"discounted by {b}%")
            else:
                t = rng.choice([5, 8, 10])
                factor *= Fraction(100 + t, 100)
                parts.append(f"increased by {t}% for sales tax")
        final = p * factor
        if (final * 100).denominator == 1:
            break
    seq = parts[0] if n == 1 else ", then ".join(parts)
    fin = f"{float(final):.2f}"
    if d >= 4:
        q = (f"The price of a {item} is {seq}. After these changes the price is ${fin}. "
             f"What was the original price? {FMT['money']}")
        ans = f"{p:.2f}"
    else:
        q = f"A {item} costs ${p}. Its price is {seq}. What is the final price? {FMT['money']}"
        ans = fin
    return Problem(q, ans, "number", "arithmetic", "percentages", d, {"dp": 2})


def _lin(coefs, vs) -> str:
    out = ""
    for c, v in zip(coefs, vs):
        if c == 0:
            continue
        mag = "" if abs(c) == 1 else str(abs(c))
        if not out:
            out = ("-" if c < 0 else "") + mag + v
        else:
            out += (" - " if c < 0 else " + ") + mag + v
    return out or "0"


def _signed(n: int) -> str:
    return f"- {abs(n)}" if n < 0 else f"+ {n}"


def _coef(c: int, term: str, first: bool = False) -> str:
    """Render c·term as it would be written by hand: 'a', '- a', '+ 3·a'."""
    mag = term if abs(c) == 1 else f"{abs(c)}·{term}"
    if first:
        return ("-" if c < 0 else "") + mag
    return ("- " if c < 0 else "+ ") + mag


def _det3(m) -> int:
    (a, b, c), (d_, e, f), (g, h, i) = m
    return a * (e * i - f * h) - b * (d_ * i - f * g) + c * (d_ * h - e * g)


def gen_linear(rng: random.Random, d: int) -> Problem:
    nz = [k for k in range(-6, 7) if k != 0]
    if d == 1:
        x, a, b = rng.randint(-12, 12), rng.randint(2, 9), rng.choice([k for k in range(-20, 21) if k])
        q, ans = f"Solve for x: {a}x {_signed(b)} = {a * x + b}.", x
    elif d == 2:
        x, a, b = rng.randint(-10, 10), rng.randint(2, 7), rng.choice(nz)
        c = rng.choice([k for k in range(2, 9) if k != a])
        e = a * (x + b) - c * x
        rhs = f"{c}x" if e == 0 else f"{c}x {_signed(e)}"
        q, ans = f"Solve for x: {a}(x {_signed(b)}) = {rhs}.", x
    elif d == 3:
        while True:
            a, b, c, e = (rng.choice(nz) for _ in range(4))
            if a * e - b * c != 0:
                break
        x, y = rng.randint(-8, 8), rng.randint(-8, 8)
        var = rng.choice("xy")
        q = (f"Solve the system: {_lin((a, b), 'xy')} = {a * x + b * y} and "
             f"{_lin((c, e), 'xy')} = {c * x + e * y}. What is the value of {var}?")
        ans = x if var == "x" else y
    elif d == 4:
        if rng.random() < 0.5:
            pc = rng.randint(4, 12)
            pa = pc + rng.randint(3, 12)
            na, nc = rng.randint(20, 150), rng.randint(20, 150)
            q = (f"A theatre sold {na + nc} tickets for a total of ${na * pa + nc * pc}. Adult tickets cost "
                 f"${pa} and child tickets cost ${pc}. How many adult tickets were sold?")
            ans = na
        else:
            m, n = rng.choice([(3, 2), (4, 2), (5, 2), (5, 3), (4, 3), (7, 3)])
            k_unit = Fraction(m - n, n - 1)
            b = rng.choice([v for v in range(4, 19) if (v * k_unit).denominator == 1])
            k = int(b * k_unit)
            p1, p2 = rng.sample(["Amara", "Jonas", "Priya", "Tariq", "Elena", "Kofi"], 2)
            q = (f"{p1} is currently {m} times as old as {p2}. In {plural(k, 'year')}, {p1} will be {n} times "
                 f"as old as {p2}. How old is {p1} now?")
            ans = m * b
    else:
        while True:
            m = [[rng.randint(-4, 4) for _ in range(3)] for _ in range(3)]
            if _det3(m) != 0 and all(any(row) for row in m):
                break
        sol = [rng.randint(-6, 6) for _ in range(3)]
        eqs = "; ".join(f"{_lin(row, 'xyz')} = {sum(c * s for c, s in zip(row, sol))}" for row in m)
        if rng.random() < 0.5:
            q, ans = f"Solve the system: {eqs}. What is the value of x + y + z?", sum(sol)
        else:
            i = rng.randrange(3)
            q, ans = f"Solve the system: {eqs}. What is the value of {'xyz'[i]}?", sol[i]
    return Problem(f"{q} {FMT['int']}", str(ans), "int", "algebra", "linear_equations", d)


def gen_number_theory(rng: random.Random, d: int) -> Problem:
    if d == 1:
        g = rng.randint(2, 18)
        a, b = g * rng.randint(2, 15), g * rng.randint(2, 15)
        if rng.random() < 0.5:
            q, ans = f"What is the greatest common divisor of {a} and {b}?", math.gcd(a, b)
        else:
            q, ans = f"What is the least common multiple of {a} and {b}?", a * b // math.gcd(a, b)
    elif d == 2:
        if rng.random() < 0.5:
            n, ans = 1, 1
            for p in rng.sample([2, 3, 5, 7], rng.randint(2, 3)):
                e = rng.randint(1, 3)
                n *= p ** e
                ans *= e + 1
            q = f"How many positive divisors does {n} have?"
        else:
            lo, hi = rng.randint(1, 60), rng.randint(120, 400)
            x, y = rng.sample([3, 4, 5, 6, 7, 9, 11], 2)
            q = f"How many integers from {lo} to {hi} inclusive are divisible by {x} or by {y}?"
            ans = sum(1 for k in range(lo, hi + 1) if k % x == 0 or k % y == 0)
    elif d == 3:
        a, b, m = rng.randint(2, 12), rng.randint(20, 120), rng.choice([7, 9, 11, 13, 17, 19, 23])
        q, ans = f"What is the remainder when {a}^{b} is divided by {m}?", pow(a, b, m)
    else:
        v = rng.choice(["crt", "zeros", "sigma"]) if d == 5 else "crt"
        if v == "crt":
            ms = rng.sample([3, 4, 5, 7, 9, 11, 13], 3 if d == 5 else 2)
            while any(math.gcd(p, r) != 1 for p, r in itertools.combinations(ms, 2)):
                ms = rng.sample([3, 4, 5, 7, 9, 11, 13], len(ms))
            rs = [rng.randint(1, m - 1) for m in ms]
            conds = ", ".join(f"remainder {r} when divided by {m}" for r, m in zip(rs, ms))
            q = f"What is the smallest positive integer that leaves {conds}?"
            ans = next(k for k in range(1, math.prod(ms) + 1) if all(k % m == r for r, m in zip(rs, ms)))
        elif v == "zeros":
            n = rng.randint(40, 600)
            q = f"How many zeros are at the end of {n}! (the factorial of {n})?"
            ans, p = 0, 5
            while p <= n:
                ans += n // p
                p *= 5
        else:
            n = rng.choice([2, 4, 8]) * rng.choice([3, 9, 5, 15]) * rng.choice([1, 7, 11])
            q = f"What is the sum of all positive divisors of {n}, including 1 and {n} itself?"
            ans = sum(k for k in range(1, n + 1) if n % k == 0)
    return Problem(f"{q} {FMT['int']}", str(ans), "int", "number_theory", "number_theory", d)


def gen_counting(rng: random.Random, d: int) -> Problem:
    kind = "int"
    if d == 1:
        if rng.random() < 0.5:
            n, k = rng.randint(6, 12), rng.randint(2, 4)
            q = f"In how many ways can a committee of {k} be chosen from {n} people?"
            ans = math.comb(n, k)
        else:
            word = "".join(rng.choice("AB") for _ in range(2)) + "".join(rng.sample("CDEF", rng.randint(2, 3)))
            word = "".join(sorted(word))
            q = f"How many distinct arrangements are there of the letters {word}?"
            ans = len(set(itertools.permutations(word)))
    elif d == 2:
        if rng.random() < 0.5:
            m, w, k = rng.randint(4, 7), rng.randint(3, 6), rng.randint(3, 4)
            j = rng.randint(1, 2)
            q = (f"A committee of {k} is chosen from {m} men and {w} women. In how many ways can this be "
                 f"done if the committee must include at least {plural(j, 'woman').replace('womans', 'women')}?")
            ans = sum(math.comb(w, i) * math.comb(m, k - i) for i in range(j, k + 1))
        else:
            s = rng.randint(4, 10)
            ev = rng.choice(["eq", "ge"])
            hits = sum(1 for a in range(1, 7) for b in range(1, 7) if (a + b == s if ev == "eq" else a + b >= s))
            what = f"exactly {s}" if ev == "eq" else f"at least {s}"
            q = f"Two fair six-sided dice are rolled. What is the probability that the sum is {what}?"
            ans, kind = Fraction(hits, 36), "fraction"
    elif d == 3:
        if rng.random() < 0.5:
            word = "AA" + "".join(rng.sample("BCDEF", rng.randint(3, 4)))
            perms = set(itertools.permutations(word))
            q = (f"How many distinct arrangements of the letters {word} are there in which the two A's "
                 "are not next to each other?")
            ans = sum(1 for p in perms if "AA" not in "".join(p))
        else:
            r, b, g, k = rng.randint(3, 6), rng.randint(2, 5), rng.randint(2, 4), rng.randint(2, 3)
            j = rng.randint(1, min(k, r))
            q = (f"A bag holds {r} red, {b} blue and {g} green marbles. {k} marbles are drawn at random "
                 f"without replacement. What is the probability that exactly {j} of them are red?")
            ans = Fraction(math.comb(r, j) * math.comb(b + g, k - j), math.comb(r + b + g, k))
            kind = "fraction"
    elif d == 4:
        v = rng.choice(["lattice", "dice3", "each"])
        if v == "lattice":
            m, n = rng.randint(4, 6), rng.randint(3, 5)
            px, py = rng.randint(1, m - 1), rng.randint(1, n - 1)
            q = (f"On a grid, a path goes from (0, 0) to ({m}, {n}) using only steps of one unit right or "
                 f"one unit up. How many such paths do not pass through the point ({px}, {py})?")
            ans = math.comb(m + n, m) - math.comb(px + py, px) * math.comb(m - px + n - py, m - px)
        elif v == "dice3":
            s = rng.randint(6, 15)
            hits = sum(1 for t in itertools.product(range(1, 7), repeat=3) if sum(t) == s)
            q = f"Three fair six-sided dice are rolled. What is the probability that the sum is exactly {s}?"
            ans, kind = Fraction(hits, 216), "fraction"
        else:
            r, b, g = rng.randint(2, 4), rng.randint(2, 4), rng.randint(2, 4)
            balls = "R" * r + "B" * b + "G" * g
            k = rng.randint(3, 4)
            combos = list(itertools.combinations(range(len(balls)), k))
            hits = sum(1 for c in combos if len({balls[i] for i in c}) == 3)
            q = (f"A bag holds {r} red, {b} blue and {g} green marbles. {k} marbles are drawn at random "
                 "without replacement. What is the probability that at least one marble of each colour is drawn?")
            ans, kind = Fraction(hits, len(combos)), "fraction"
    else:
        v = rng.choice(["derange", "digits"])
        if v == "derange":
            n = rng.randint(4, 6)
            q = (f"{n} letters are written to {n} different people and {n} addressed envelopes are prepared. "
                 "In how many ways can the letters be placed, one per envelope, so that no letter is in its "
                 "correct envelope?")
            ans = sum(1 for p in itertools.permutations(range(n)) if all(p[i] != i for i in range(n)))
        else:
            prop = rng.choice(["distinct_even", "increasing", "sum", "distinct_div5"])
            s = rng.randint(8, 28)
            desc = {"distinct_even": "are even and have four different digits",
                    "increasing": "have digits that strictly increase from left to right",
                    "sum": f"have digits that add up to {s}",
                    "distinct_div5": "are divisible by 5 and have four different digits"}[prop]

            def ok(n: int) -> bool:
                ds = [int(c) for c in str(n)]
                if prop == "distinct_even":
                    return n % 2 == 0 and len(set(ds)) == 4
                if prop == "increasing":
                    return all(a < b for a, b in zip(ds, ds[1:]))
                if prop == "sum":
                    return sum(ds) == s
                return n % 5 == 0 and len(set(ds)) == 4

            q = f"How many four-digit positive integers (1000 to 9999) {desc}?"
            ans = sum(1 for n in range(1000, 10000) if ok(n))
    if kind == "fraction":
        return Problem(f"{q} {FMT['fraction']}", frac_str(ans), "fraction",
                       "combinatorics_probability", "probability", d)
    return Problem(f"{q} {FMT['int']}", str(ans), "int", "combinatorics_probability", "counting", d)


def gen_recurrence(rng: random.Random, d: int) -> Problem:
    while True:
        if d <= 2:
            p, r, t, k = rng.randint(1, 6), rng.choice([2, 3, -2]), rng.randint(-5, 7), rng.randint(4, 5) + d
            seq = [p]
            for _ in range(k - 1):
                seq.append(r * seq[-1] + t)
            rule = (f"a_1 = {p} and a_n = {_coef(r, 'a_(n-1)', True)}" + (f" {_signed(t)}" if t else "") + " for n ≥ 2")
            q, ans = f"A sequence is defined by {rule}. What is a_{k}?", seq[-1]
        elif d == 3:
            p, q0 = rng.randint(1, 5), rng.randint(1, 5)
            r, s, k = rng.choice([1, 2, 3]), rng.choice([1, -1, 2]), rng.randint(6, 9)
            seq = [p, q0]
            for _ in range(k - 2):
                seq.append(r * seq[-1] + s * seq[-2])
            q = (f"A sequence is defined by a_1 = {p}, a_2 = {q0} and a_n = {_coef(r, 'a_(n-1)', True)} "
                 f"{_coef(s, 'a_(n-2)')} for n ≥ 3. What is a_{k}?")
            ans = seq[-1]
        elif d == 4:
            p, q0, k = rng.randint(1, 4), rng.randint(1, 4), rng.randint(6, 8)
            seq = [p, q0]
            for n in range(3, k + 1):
                seq.append(n * seq[-1] - seq[-2])
            q = (f"A sequence is defined by a_1 = {p}, a_2 = {q0} and a_n = n·a_(n-1) - a_(n-2) for n ≥ 3. "
                 f"What is a_{k}?")
            ans = seq[-1]
        else:
            if rng.random() < 0.5:
                m = rng.randint(6, 40)
                n, steps = m, 0
                while n != 1:
                    n = n // 2 if n % 2 == 0 else 3 * n + 1
                    steps += 1
                q = (f"Start with the number {m}. Repeatedly apply this rule: if the number is even, halve it; "
                     "if it is odd, multiply it by 3 and add 1. How many applications of the rule does it take "
                     "to reach 1 for the first time?")
                ans = steps
                if not 6 <= steps <= 24:
                    continue
            else:
                p, q0, r, s, k = rng.randint(1, 4), rng.randint(1, 4), rng.choice([1, 2]), rng.choice([1, -1]), \
                    rng.randint(7, 9)
                seq = [p, q0]
                for _ in range(k - 2):
                    seq.append(r * seq[-1] + s * seq[-2])
                q = (f"A sequence is defined by a_1 = {p}, a_2 = {q0} and a_n = {_coef(r, 'a_(n-1)', True)} "
                     f"{_coef(s, 'a_(n-2)')} for n ≥ 3. What is the sum of the first {k} terms?")
                ans = sum(seq)
        if abs(ans) < 10 ** 7:
            return Problem(f"{q} {FMT['int']}", str(ans), "int", "algebra", "recurrences", d)


def _expr(rng: random.Random, depth: int) -> tuple[str, int]:
    if depth == 0:
        v = rng.randint(2, 12)
        return str(v), v
    op = rng.choice("++--***//^")
    if op == "^":
        b, e = rng.randint(2, 5), rng.choice([2, 2, 3])
        return f"{b}^{e}", b ** e
    ls, lv = _expr(rng, depth - 1)
    rs, rv = _expr(rng, depth - 1)
    if op == "/":
        if rv == 0 or abs(rv) > 30:
            op = "+"
        else:
            quo = rng.randint(2, 9)
            num = rv * quo
            return f"({num if num >= 0 else f'({num})'} ÷ {rs})", quo
    if op == "*" and abs(lv * rv) > 5000:
        op = "-"
    if op == "+":
        return f"({ls} + {rs})", lv + rv
    if op == "-":
        return f"({ls} - {rs})", lv - rv
    return f"({ls} × {rs})", lv * rv


def _to_base(n: int, b: int) -> str:
    digits = "0123456789ABCDEF"
    out = ""
    while n:
        out = digits[n % b] + out
        n //= b
    return out or "0"


def gen_expression(rng: random.Random, d: int) -> Problem:
    if d in (3, 5) and rng.random() < 0.4:
        k = 2 if d == 3 else 3
        vals = [rng.randint(20, 400) for _ in range(k)]
        bases = [rng.choice([2, 3, 5, 8, 16]) for _ in range(k)]
        signs = ["+"] + [rng.choice("+-") for _ in range(k - 1)]
        text = " ".join((f"{s} " if i else "") + f"{_to_base(v, b)} (base {b})"
                        for i, (v, b, s) in enumerate(zip(vals, bases, signs)))
        ans = sum(v if s == "+" else -v for v, s in zip(vals, signs))
        return Problem(f"Compute {text}. Give the result as a whole number in base 10.", str(ans), "int",
                       "arithmetic", "base_conversion", d)
    depth = 1 if d == 1 else 2 if d <= 3 else 3
    s, v = _expr(rng, depth)
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    note = " Here ^ means exponentiation." if "^" in s else ""
    return Problem(f"Evaluate: {s}.{note} {FMT['int']}", str(v), "int", "arithmetic", "order_of_operations", d)


_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = [("January", 31), ("February", 28), ("March", 31), ("April", 30), ("May", 31), ("June", 30),
           ("July", 31), ("August", 31), ("September", 30), ("October", 31), ("November", 30), ("December", 31)]


def _hm(minutes: int) -> str:
    minutes %= 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _dur(m: int) -> str:
    h, r = divmod(m, 60)
    if h and r:
        return f"{plural(h, 'hour')} {plural(r, 'minute')}"
    return plural(h, "hour") if h else plural(r, "minute")


def gen_time(rng: random.Random, d: int) -> Problem:
    v = rng.choice(["sessions", "weekday", "flight"])
    if v == "sessions":
        while True:
            start = rng.randrange(7 * 60, 12 * 60, 5)
            segs = [rng.randrange(20, 130, 5) for _ in range(d + 1)]
            if start + sum(segs) < 24 * 60:
                break
        names = ["a session", "a break"]
        parts = ", then ".join(f"{names[i % 2]} of {_dur(m)}" for i, m in enumerate(segs))
        q = (f"A workshop starts at {_hm(start)}. The schedule is {parts}, with no gaps in between. "
             f"At what time does the last part end? {FMT['time']}")
        return Problem(q, _hm(start + sum(segs)), "time", "spatial_temporal", "clock_arithmetic", d)
    if v == "weekday":
        w0 = rng.randrange(7)
        if d == 1:
            n = rng.randint(10, 500)
            q = f"Today is a {_WEEKDAYS[w0]}. What day of the week will it be {n} days from today?"
            ans = _WEEKDAYS[(w0 + n) % 7]
        elif d <= 3:
            a, b = sorted(rng.sample(range(1, 31), 2))
            if rng.random() < 0.5:
                a, b = b, a
            q = (f"The {ordinal(a)} of a certain month is a {_WEEKDAYS[w0]}. What day of the week is the "
                 f"{ordinal(b)} of the same month?")
            ans = _WEEKDAYS[(w0 + b - a) % 7]
        else:
            m1, m2 = sorted(rng.sample(range(12), 2))
            d1, d2 = rng.randint(1, _MONTHS[m1][1]), rng.randint(1, _MONTHS[m2][1])
            doy = lambda m, day: sum(n for _, n in _MONTHS[:m]) + day  # noqa: E731
            q = (f"In a year that is not a leap year, {_MONTHS[m1][0]} {d1} falls on a {_WEEKDAYS[w0]}. "
                 f"What day of the week is {_MONTHS[m2][0]} {d2} of that year?")
            ans = _WEEKDAYS[(w0 + doy(m2, d2) - doy(m1, d1)) % 7]
        return Problem(f"{q} Answer with the day name only.", ans, "text", "spatial_temporal",
                       "calendar_arithmetic", d, {"distractors": [w for w in _WEEKDAYS if w != ans]})
    dep = rng.randrange(0, 1440, 5)
    f1, tz1 = rng.randrange(95, 780, 5), rng.choice([-7, -5, -3, -2, 2, 3, 4, 6, 8])
    rel = lambda tz: f"{plural(abs(tz), 'hour')} {'ahead of' if tz > 0 else 'behind'}"  # noqa: E731
    if d <= 3:
        q = (f"A flight departs at {_hm(dep)} local time and lasts {_dur(f1)}. The destination's clocks are "
             f"{rel(tz1)} the departure city's. What is the local time at the destination when the flight lands?")
        ans = dep + f1 + tz1 * 60
    else:
        lay, f2, tz2 = rng.randrange(45, 240, 5), rng.randrange(60, 600, 5), rng.choice([-4, -2, -1, 1, 2, 5])
        q = (f"A traveller's first flight departs at {_hm(dep)} local time and lasts {_dur(f1)}. The stopover "
             f"city's clocks are {rel(tz1)} the departure city's. After a layover of {_dur(lay)}, a second flight "
             f"lasting {_dur(f2)} goes to the final destination, whose clocks are {rel(tz2)} the stopover city's. "
             "What is the local time at the final destination on arrival?")
        ans = dep + f1 + tz1 * 60 + lay + f2 + tz2 * 60
    return Problem(f"{q} {FMT['time']}", _hm(ans), "time", "spatial_temporal", "time_zones", d)


def gen_geometry(rng: random.Random, d: int) -> Problem:
    v = rng.choice({1: ["triangle", "path"], 2: ["path", "lshape"], 3: ["lshape", "box"],
                    4: ["box", "tiles"], 5: ["tiles", "cubes"]}[d])
    if v == "triangle":
        a, b, c = rng.choice([(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)])
        k = rng.randint(1, 6)
        a, b, c = a * k, b * k, c * k
        what = rng.choice(["hyp", "perim", "area"])
        q = f"A right triangle has legs of length {a} cm and {b} cm. "
        q += {"hyp": "How long is its hypotenuse in cm?", "perim": "What is its perimeter in cm?",
              "area": "What is its area in square cm?"}[what]
        ans = {"hyp": c, "perim": a + b + c, "area": a * b // 2}[what]
    elif v == "path":
        a, b, w = rng.randint(8, 40), rng.randint(6, 30), rng.randint(1, 4)
        q = (f"A rectangular garden measures {a} m by {b} m. A path {w} m wide runs all the way around the "
             "outside of it. What is the area of the path in square metres?")
        ans = (a + 2 * w) * (b + 2 * w) - a * b
    elif v == "lshape":
        a, b = rng.randint(8, 30), rng.randint(8, 30)
        c, e = rng.randint(2, a - 3), rng.randint(2, b - 3)
        if rng.random() < 0.5:
            q = (f"An L-shaped room is made by cutting a {c} m by {e} m rectangle out of one corner of a "
                 f"{a} m by {b} m rectangle. What is the floor area of the room in square metres?")
            ans = a * b - c * e
        else:
            q = (f"An L-shaped room is made by cutting a {c} m by {e} m rectangle out of one corner of a "
                 f"{a} m by {b} m rectangle. What is the perimeter of the room in metres?")
            ans = 2 * (a + b)
    elif v == "box":
        l, w, h = rng.randint(3, 12), rng.randint(3, 12), rng.randint(2, 10)
        if rng.random() < 0.5:
            q = (f"A closed rectangular box is {l} cm long, {w} cm wide and {h} cm high. What is its total "
                 "surface area in square cm?")
            ans = 2 * (l * w + l * h + w * h)
        else:
            q = (f"An open-topped rectangular tank is {l} m long, {w} m wide and {h} m deep. Its inside walls "
                 "and floor are to be painted. What area is painted, in square metres?")
            ans = l * w + 2 * h * (l + w)
    elif v == "tiles":
        s = rng.choice([20, 25, 50])
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        per_box, price = rng.choice([8, 10, 12, 15]), rng.randint(12, 45)
        tiles = (a * 100 // s) * (b * 100 // s)
        boxes = -(-tiles // per_box)
        q = (f"A floor measuring {a} m by {b} m is covered with square tiles {s} cm on a side. Tiles are sold "
             f"only in full boxes of {per_box}, and each box costs ${price}. What is the total cost of the "
             "boxes needed, in dollars?")
        ans = boxes * price
    else:
        l, w, h = rng.randint(3, 8), rng.randint(3, 8), rng.randint(3, 8)
        faces = rng.choice([1, 2, 0])
        q = (f"A solid block measuring {l} by {w} by {h} unit cubes is painted on all six outer faces and then "
             f"cut into unit cubes. How many of the unit cubes have paint on exactly {plural(faces, 'face')}?")
        ans = {2: 4 * ((l - 2) + (w - 2) + (h - 2)),
               1: 2 * ((l - 2) * (w - 2) + (l - 2) * (h - 2) + (w - 2) * (h - 2)),
               0: (l - 2) * (w - 2) * (h - 2)}[faces]
    return Problem(f"{q} {FMT['int']}", str(ans), "int", "geometry", "measurement", d)


GENERATORS = [gen_inventory, gen_rates, gen_percent, gen_linear, gen_number_theory, gen_counting,
              gen_recurrence, gen_expression, gen_time, gen_geometry]
