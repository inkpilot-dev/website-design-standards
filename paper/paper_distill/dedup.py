"""Near-duplicate detection for teacher-written questions (MinHash over word shingles)."""
from __future__ import annotations

import re
import zlib

import numpy as np

_PRIME = np.uint64(4294967311)


def _shingles(text: str, k: int) -> set[int]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < k:
        return {zlib.crc32(" ".join(words).encode())}
    return {zlib.crc32(" ".join(words[i:i + k]).encode()) for i in range(len(words) - k + 1)}


def near_duplicates(texts: list[str], threshold: float = 0.6, k: int = 3, perms: int = 64, bands: int = 16,
                    seed: int = 0) -> set[int]:
    """Indices to drop so that no two kept texts have shingle Jaccard similarity >= threshold.

    The earlier text of a similar pair is kept. Candidate pairs come from LSH banding and
    are then confirmed with the exact Jaccard similarity, so nothing is dropped on a hash collision.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2 ** 31, perms, dtype=np.uint64)[:, None]
    b = rng.integers(0, 2 ** 31, perms, dtype=np.uint64)[:, None]
    sets = [_shingles(t, k) for t in texts]
    rows = perms // bands
    buckets: dict[tuple, list[int]] = {}
    drop: set[int] = set()
    for i, s in enumerate(sets):
        h = np.fromiter(s, dtype=np.uint64)[None, :]
        sig = ((a * h + b) % _PRIME).min(axis=1)
        keys = [(j, sig[j * rows:(j + 1) * rows].tobytes()) for j in range(bands)]
        cands = {c for key in keys for c in buckets.get(key, [])}
        if any(len(s & sets[c]) / len(s | sets[c]) >= threshold for c in cands):
            drop.add(i)
            continue
        for key in keys:
            buckets.setdefault(key, []).append(i)
    return drop
