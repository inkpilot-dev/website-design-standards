from engine.stats import mcnemar_exact, paired_bootstrap, wilson


def test_wilson_known_values():
    lo, hi = wilson(0, 95)
    assert lo == 0.0 and abs(hi - 0.0389) < 1e-3
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3


def test_mcnemar_exact():
    # b=10, c=2: 2 * (C(12,0)+C(12,1)+C(12,2)) / 2^12 = 158/4096
    assert abs(mcnemar_exact(10, 2) - 158 / 4096) < 1e-12
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(7, 0) == 2 * 0.5**7


def test_paired_bootstrap_bounds():
    base = [False] * 100
    cand = [True] * 20 + [False] * 80
    mean, lo, hi = paired_bootstrap(base, cand, iters=2000, seed=1)
    assert abs(mean - 20.0) < 1e-9 and 10 < lo < 20 < hi < 30
