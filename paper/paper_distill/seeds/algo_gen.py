"""Code tracing, graph, scheduling, string and search-style generators."""
from __future__ import annotations

import contextlib
import heapq
import io
import random
from collections import deque

from .base import FMT, Problem, plural

_SAFE = {"range": range, "len": len, "print": print, "sum": sum, "min": min, "max": max, "sorted": sorted,
         "abs": abs, "str": str, "int": int, "list": list, "dict": dict, "enumerate": enumerate}


def _run(code: str) -> str:
    """Run one of our own templates (never model output) and capture what it prints."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {"__builtins__": _SAFE})  # noqa: S102 - the code is generated from the templates below
    return buf.getvalue().strip()


def gen_trace(rng: random.Random, d: int) -> Problem:
    r = rng.randint
    t = rng.choice({1: ["acc", "str"], 2: ["acc", "list", "pair"], 3: ["list", "nested", "collatz"],
                    4: ["nested", "dict", "rec"], 5: ["rec", "dict", "mixed"]}[d])
    if t == "acc":
        code = (f"total = {r(0, 5)}\nfor i in range({r(1, 4)}, {r(8, 14)}):\n    if i % {r(2, 4)} == {r(0, 1)}:\n"
                f"        total += i * {r(2, 5)}\n    else:\n        total -= {r(1, 4)}\nprint(total)")
    elif t == "str":
        w = rng.choice(["planet", "garden", "silver", "bridge", "candle", "forest"])
        code = (f's = "{w}"\nout = ""\nfor i, ch in enumerate(s):\n    if i % 2 == 0:\n        out = ch + out\n'
                "    else:\n        out = out + ch\nprint(out)")
    elif t == "pair":
        code = (f"a, b = {r(1, 6)}, {r(1, 6)}\nfor _ in range({r(4, 7)}):\n    a, b = b, (a + b) % {r(7, 13)}\n"
                "print(a, b)")
    elif t == "list":
        xs = [r(1, 20) for _ in range(r(5, 7))]
        tail = rng.choice(["print(sum(ys))", "print(ys)", "ys.sort()\nprint(ys[1:-1])"])
        code = (f"xs = {xs}\nys = []\nfor x in xs:\n    if x % 2 == 0:\n        ys.append(x // 2)\n    else:\n"
                f"        ys.insert(0, x * 3 + 1)\n{tail}")
    elif t == "nested":
        code = (f"count = 0\nfor i in range(1, {r(5, 8)}):\n    for j in range(i, {r(6, 9)}):\n"
                f"        if (i * j) % {r(2, 4)} == 0:\n            count += 1\nprint(count)")
    elif t == "collatz":
        code = (f"n = {r(6, 30)}\nsteps = 0\nwhile n != 1:\n    if n % 2 == 0:\n        n = n // 2\n    else:\n"
                "        n = 3 * n + 1\n    steps += 1\nprint(steps)")
    elif t == "dict":
        w = rng.choice(["banana bandana", "mississippi river", "committee meeting", "pepper and paper"])
        code = (f'text = "{w}"\ncounts = {{}}\nfor ch in text:\n    if ch != " ":\n'
                "        counts[ch] = counts.get(ch, 0) + 1\nbest = sorted(counts, key=lambda k: (-counts[k], k))[0]\n"
                "print(best, counts[best], len(counts))")
    elif t == "rec":
        code = (f"def f(n):\n    if n < 2:\n        return n + {r(1, 3)}\n    return f(n - 1) + {r(1, 3)} * f(n - 2)\n"
                f"print(f({r(5, 8)}))")
    else:
        xs = [r(1, 9) for _ in range(r(6, 8))]
        code = (f"xs = {xs}\nstack = []\nfor x in xs:\n    while stack and stack[-1] < x:\n        stack.pop()\n"
                "    stack.append(x)\nprint(stack, len(stack))")
    q = f"What does this Python program print?\n\n```python\n{code}\n```\nGive exactly the printed output."
    return Problem(q, _run(code), "exact", "code_tracing", "program_tracing", d)


def _dijkstra(adj, src):
    dist, ways = {src: 0}, {src: 1}
    pq = [(0, src)]
    while pq:
        du, u = heapq.heappop(pq)
        if du > dist[u]:
            continue
        for v, w in adj[u]:
            nd = du + w
            if nd < dist.get(v, float("inf")):
                dist[v], ways[v] = nd, ways[u]
                heapq.heappush(pq, (nd, v))
            elif nd == dist[v]:
                ways[v] += ways[u]
    return dist, ways


def gen_graph(rng: random.Random, d: int) -> Problem:
    n = 4 + d if d <= 3 else 8
    nodes = [chr(65 + i) for i in range(n)]
    while True:
        edges = {}
        for i in range(1, n):  # random spanning tree keeps the graph connected
            edges[(nodes[rng.randrange(i)], nodes[i])] = rng.randint(1, 15)
        for _ in range(n // 2 + d):
            a, b = sorted(rng.sample(nodes, 2))
            edges.setdefault((a, b), rng.randint(1, 15))
        adj = {u: [] for u in nodes}
        for (a, b), w in edges.items():
            adj[a].append((b, w))
            adj[b].append((a, w))
        s, t = rng.sample(nodes, 2)
        if (min(s, t), max(s, t)) in edges:
            continue  # require at least one intermediate decision
        dist, ways = _dijkstra(adj, s)
        want_path = d >= 4 and rng.random() < 0.5
        if not want_path or ways[t] == 1:
            break
    listing = ", ".join(f"{a}-{b} ({w})" for (a, b), w in sorted(edges.items()))
    body = (f"A network has towns {', '.join(nodes)}. Two-way roads and their lengths in km are: {listing}.")
    if want_path:
        prev_dist, _ = _dijkstra(adj, t)
        path, cur = [s], s
        while cur != t:
            cur = next(v for v, w in adj[cur] if dist[cur] + w + prev_dist[v] == dist[t])
            path.append(cur)
        return Problem(f"{body} What is the shortest route from {s} to {t}? List the towns visited in order, "
                       "separated by commas.", ", ".join(path), "list", "algorithmic_planning", "shortest_path", d)
    return Problem(f"{body} What is the length in km of the shortest route from {s} to {t}? {FMT['int']}",
                   str(dist[t]), "int", "algorithmic_planning", "shortest_path", d)


def gen_schedule(rng: random.Random, d: int) -> Problem:
    n = 3 + d
    tasks = [chr(65 + i) for i in range(n)]
    dur = {t: rng.randint(1, 9) for t in tasks}
    pre = {t: sorted(rng.sample(tasks[:i], rng.randint(0, min(i, 2)))) for i, t in enumerate(tasks)}
    finish: dict[str, int] = {}
    for t in tasks:
        finish[t] = dur[t] + max((finish[p] for p in pre[t]), default=0)
    total = max(finish.values())
    lines = "; ".join(f"{t} takes {plural(dur[t], 'day')}" +
                      (f" and needs {' and '.join(pre[t])} finished first" if pre[t] else " and has no prerequisites")
                      for t in tasks)
    body = (f"A project has these tasks: {lines}. Any number of tasks can run at the same time once their "
            "prerequisites are finished.")
    if d >= 4 and rng.random() < 0.5:
        tail = {t: dur[t] for t in tasks}  # longest path from the start of t to the end of the project
        for t in reversed(tasks):
            succ = [s for s in tasks if t in pre[s]]
            tail[t] = dur[t] + max((tail[s] for s in succ), default=0)
        t = rng.choice(tasks)
        return Problem(f"{body} The project starts on day 0 and must finish as early as possible. What is the "
                       f"latest day on which task {t} can start without delaying the project? {FMT['int']}",
                       str(total - tail[t]), "int", "algorithmic_planning", "scheduling", d)
    return Problem(f"{body} What is the minimum number of days needed to finish the whole project? {FMT['int']}",
                   str(total), "int", "algorithmic_planning", "scheduling", d)


_WORDS = ["planet", "garden", "window", "bridge", "silver", "market", "candle", "forest", "pocket", "harbor",
          "lantern", "compass", "journey", "thunder", "blanket", "village", "kitchen", "diamond", "captain",
          "library", "mountain", "notebook", "sandwich", "triangle", "keyboard", "painting", "umbrella"]


def gen_string(rng: random.Random, d: int) -> Problem:
    while True:
        w = rng.choice(_WORDS)
        s, ops = w, []
        for _ in range(d + 1):
            op = rng.choice(["rev", "rot", "caesar", "vowels", "ends", "third", "sort"])
            if op == "rev":
                s, text = s[::-1], "reverse the string"
            elif op == "rot":
                k = rng.randint(1, 3)
                s, text = s[k:] + s[:k], f"move the first {plural(k, 'letter')} to the end"
            elif op == "caesar":
                k = rng.randint(1, 4)
                s = "".join(chr((ord(c) - 97 + k) % 26 + 97) for c in s)
                text = f"replace every letter with the letter {plural(k, 'place')} later in the alphabet (z wraps to a)"
            elif op == "vowels":
                s, text = "".join(c for c in s if c not in "aeiou"), "delete every vowel (a, e, i, o, u)"
            elif op == "ends":
                s = s[-1] + s[1:-1] + s[0] if len(s) > 1 else s
                text = "swap the first and last letters"
            elif op == "third":
                s = "".join(c for i, c in enumerate(s, 1) if i % 3)
                text = "delete every third letter (positions 3, 6, 9 and so on)"
            else:
                s, text = "".join(sorted(s)), "sort the letters alphabetically"
            ops.append(text)
        if len(s) >= 3 and len(set(ops)) == len(ops):
            break
    steps = "; ".join(f"({i + 1}) {t}" for i, t in enumerate(ops))
    return Problem(f'Start with the word "{w}" and apply these steps in order: {steps}. What is the final string? '
                   "Answer with the string only, in lowercase.", s, "word", "state_tracking", "string_operations", d)


def _jug_steps(a: int, b: int, target: int) -> int | None:
    seen, q = {(0, 0)}, deque([(0, 0, 0)])
    while q:
        x, y, n = q.popleft()
        if target in (x, y):
            return n
        pour_ab, pour_ba = min(x, b - y), min(y, a - x)
        for nx, ny in [(a, y), (x, b), (0, y), (x, 0), (x - pour_ab, y + pour_ab), (x + pour_ba, y - pour_ba)]:
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny, n + 1))
    return None


def gen_jugs(rng: random.Random, d: int) -> Problem:
    lo, hi = {1: (2, 3), 2: (4, 4), 3: (5, 6), 4: (7, 8), 5: (9, 14)}[d]
    while True:
        a, b = sorted(rng.sample(range(3, 14), 2))
        t = rng.randint(1, b - 1)
        if t in (a, b):
            continue
        steps = _jug_steps(a, b, t)
        if steps is not None and lo <= steps <= hi:
            break
    q = (f"You have an unmarked {a}-litre jug, an unmarked {b}-litre jug and an unlimited water supply. In one "
         "move you can fill a jug completely, empty a jug completely, or pour from one jug into the other until "
         f"the first is empty or the second is full. Both jugs start empty. What is the smallest number of moves "
         f"after which one of the jugs holds exactly {plural(t, 'litre')}? {FMT['int']}")
    return Problem(q, str(steps), "int", "algorithmic_planning", "state_space_search", d)


def gen_coins(rng: random.Random, d: int) -> Problem:
    coins = sorted({1} | set(rng.sample(range(3, 16), 2 if d <= 3 else 3)))
    n = rng.randint(15, 30) + 12 * d
    names = ", ".join(str(c) for c in coins)
    if d >= 4 and rng.random() < 0.5:
        n = rng.randint(12, 30)
        ways = [1] + [0] * n
        for c in coins:
            for v in range(c, n + 1):
                ways[v] += ways[v - c]
        q = (f"A country has coins worth {names}. In how many different ways can a total of {n} be made, where "
             "two ways are the same if they use the same number of each coin?")
        return Problem(f"{q} {FMT['int']}", str(ways[n]), "int", "algorithmic_planning", "dynamic_programming", d)
    best = [0] + [10 ** 9] * n
    for v in range(1, n + 1):
        best[v] = 1 + min(best[v - c] for c in coins if c <= v)
    q = f"A country has coins worth {names}. What is the smallest number of coins that add up to exactly {n}?"
    return Problem(f"{q} {FMT['int']}", str(best[n]), "int", "algorithmic_planning", "dynamic_programming", d)


GENERATORS = [gen_trace, gen_graph, gen_schedule, gen_string, gen_jugs, gen_coins]
