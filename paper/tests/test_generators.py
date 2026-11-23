import itertools
import math
import random
from fractions import Fraction

import pytest

from paper_distill import seeds
from paper_distill.seeds import algo_gen, logic_gen, math_gen
from paper_distill.verify import check_answer

TYPES = {"int", "fraction", "number", "text", "yesno", "list", "set", "exact", "word", "time"}


@pytest.mark.parametrize("gen", seeds.GENERATORS, ids=lambda g: g.__name__)
def test_generator_contract(gen):
    """Every generator, at every difficulty: well-formed, deterministic, and its own answer passes the checker."""
    for d in range(1, 6):
        for s in range(40):
            p = seeds.make_problem(gen, random.Random(1000 * d + s), d)
            again = seeds.make_problem(gen, random.Random(1000 * d + s), d)
            assert (p.question, p.answer) == (again.question, again.answer)
            assert p.answer_type in TYPES and p.difficulty == d and len(p.question) > 30
            assert check_answer(p.answer_type, p.answer, p.answer, p.meta), (gen.__name__, p.answer)
            assert not check_answer(p.answer_type, p.answer, "", p.meta)
            for wrong in p.meta.get("distractors", []):
                assert not check_answer(p.answer_type, p.answer, wrong, p.meta)


def test_generate_is_distinct_and_deterministic():
    a, b = seeds.generate(600, 7), seeds.generate(600, 7)
    assert a == b and len({r["id"] for r in a}) == 600
    held_out = seeds.generate(200, 8, exclude={r["id"] for r in a})
    assert not {r["id"] for r in held_out} & {r["id"] for r in a}
    assert len({r["generator"] for r in a}) == len(seeds.GENERATORS)


def test_jug_search_matches_known_cases():
    assert algo_gen._jug_steps(3, 5, 4) == 6
    assert algo_gen._jug_steps(3, 5, 3) == 1
    assert algo_gen._jug_steps(4, 6, 3) is None


def test_ordering_puzzles_have_one_solution():
    """Re-solve each puzzle from its text alone, independently of the generator's bookkeeping."""
    import re
    for s in range(40):
        p = logic_gen.gen_ordering(random.Random(s), 1 + s % 5)
        names = re.search(r"\(([^)]*)\)", p.question).group(1).split(", ")
        sols = []
        for perm in itertools.permutations(names):
            pos = {n: i for i, n in enumerate(perm)}
            ok = True
            for a, b in re.findall(r"(\w+) is immediately to the left of (\w+)\.", p.question):
                ok &= pos[b] - pos[a] == 1
            for a, b in re.findall(r"(\w+) is somewhere to the left of (\w+)\.", p.question):
                ok &= pos[a] < pos[b]
            for g, a, b in re.findall(r"exactly (\d) pe\w+ between (\w+) and (\w+)\.", p.question):
                ok &= abs(pos[a] - pos[b]) == int(g) + 1
            for a, b in re.findall(r"(\w+) is not next to (\w+)\.", p.question):
                ok &= abs(pos[a] - pos[b]) != 1
            for a in re.findall(r"(\w+) is at one of the two ends\.", p.question):
                ok &= pos[a] in (0, len(names) - 1)
            for a in re.findall(r"(\w+) is not at either end\.", p.question):
                ok &= pos[a] not in (0, len(names) - 1)
            for a, k in re.findall(r"(\w+) is in position (\d)\.", p.question):
                ok &= pos[a] == int(k) - 1
            if ok:
                sols.append(perm)
        assert len(sols) == 1, p.question
        if p.answer_type == "list":
            assert ", ".join(sols[0]) == p.answer
        else:
            k = int(re.search(r"Who is in position (\d)", p.question).group(1))
            assert sols[0][k - 1] == p.answer


def test_spot_checks_against_independent_maths():
    for s in range(60):
        rng = random.Random(s)
        p = math_gen.gen_number_theory(rng, 3)
        import re
        a, b, m = map(int, re.search(r"when (\d+)\^(\d+) is divided by (\d+)", p.question).groups())
        assert int(p.answer) == (a ** b) % m
    for s in range(60):
        p = math_gen.gen_rates(random.Random(s), 1)
        if p.question.startswith("Pipe A can fill"):
            import re
            a, b = map(int, re.findall(r"in (\d+) hours", p.question))
            assert Fraction(p.answer) == Fraction(a * b, a + b)
    for s in range(30):
        p = math_gen.gen_counting(random.Random(s), 1)
        if "committee" in p.question:
            import re
            k, n = map(int, re.search(r"committee of (\d+) be chosen from (\d+)", p.question).groups())
            assert int(p.answer) == math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
