"""Statistics used by the Before-vs-After gate: Wilson CI, exact McNemar, paired bootstrap."""

from __future__ import annotations

import math
import random
from typing import Sequence


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b = candidate-only passes, c = baseline-only passes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def discordant(base: Sequence[bool], cand: Sequence[bool]) -> tuple[int, int]:
    b = sum(1 for x, y in zip(base, cand) if y and not x)
    c = sum(1 for x, y in zip(base, cand) if x and not y)
    return b, c


def paired_bootstrap(base: Sequence[bool], cand: Sequence[bool], iters: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean difference (cand - base) in percentage points and its 95% percentile interval."""
    if len(base) != len(cand):
        raise ValueError("paired samples must have equal length")
    n = len(base)
    if n == 0:
        return (0.0, 0.0, 0.0)
    d = [int(y) - int(x) for x, y in zip(base, cand)]
    mean = 100.0 * sum(d) / n
    try:
        import numpy as np  # noqa: WPS433

        rng = np.random.default_rng(seed)
        arr = np.asarray(d, dtype=np.int8)
        idx = rng.integers(0, n, size=(iters, n))
        means = 100.0 * arr[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return (mean, float(lo), float(hi))
    except ImportError:
        rng2 = random.Random(seed)
        means = sorted(100.0 * sum(rng2.choices(d, k=n)) / n for _ in range(iters))
        return (mean, means[int(0.025 * iters)], means[int(0.975 * iters) - 1])
