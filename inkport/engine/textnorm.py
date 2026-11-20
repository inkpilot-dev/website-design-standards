"""Text normalization, hashing, near-duplicate detection and soft-match metrics."""

from __future__ import annotations

import hashlib
import hmac
import re
import string
import unicodedata
from collections import Counter

_WS = re.compile(r"\s+")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", text or "").casefold()).strip()


def normalize_loose(text: str) -> str:
    """normalize() plus punctuation removal; used for answer matching."""
    return _WS.sub(" ", normalize(text).translate(_PUNCT)).strip()


def prompt_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def split_fraction(salt: str, group: str) -> float:
    """Deterministic value in [0,1) per group: HMAC-SHA256(salt, group)[0:8] / 2^64."""
    digest = hmac.new(salt.encode("utf-8"), group.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


# ---------------------------------------------------------------- MinHash / LSH -----------

_PERMS = 64
_BANDS = 16  # 16 bands x 4 rows: candidate pairs at Jaccard ~0.5+, verified at >= 0.8 below
_ROWS = _PERMS // _BANDS


def shingles(text: str, k: int = 5) -> set[str]:
    t = normalize_loose(text)
    if len(t) <= k:
        return {t}
    return {t[i : i + k] for i in range(len(t) - k + 1)}


_MERSENNE = (1 << 61) - 1
_COEFFS = [
    (
        int.from_bytes(hashlib.sha256(b"a%d" % i).digest()[:8], "little") % _MERSENNE | 1,
        int.from_bytes(hashlib.sha256(b"b%d" % i).digest()[:8], "little") % _MERSENNE,
    )
    for i in range(_PERMS)
]


def minhash(sh: set[str]) -> list[int]:
    """64-permutation MinHash using universal hashing (a*h + b) mod (2^61 - 1)."""
    base = [int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little") for s in sh]
    return [min((a * h + b) % _MERSENNE for h in base) for a, b in _COEFFS]


def est_jaccard(a: list[int], b: list[int]) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def near_duplicate_groups(texts: list[str], threshold: float = 0.8) -> list[int]:
    """Union-find cluster id per text; texts with estimated Jaccard >= threshold share a cluster."""
    n = len(texts)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sigs = [minhash(shingles(t)) for t in texts]
    buckets: dict[tuple, list[int]] = {}
    for idx, sig in enumerate(sigs):
        for b in range(_BANDS):
            key = (b, tuple(sig[b * _ROWS : (b + 1) * _ROWS]))
            buckets.setdefault(key, []).append(idx)
    for members in buckets.values():
        if len(members) < 2:
            continue
        head = members[0]
        for other in members[1:]:
            if find(head) != find(other) and est_jaccard(sigs[head], sigs[other]) >= threshold:
                parent[find(other)] = find(head)
    return [find(i) for i in range(n)]


# ---------------------------------------------------------------- soft metrics ------------


def token_f1(pred: str, ref: str) -> float:
    p = normalize_loose(pred).split()
    r = normalize_loose(ref).split()
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(r)
    return 2 * precision * recall / (precision + recall)


def word_count(text: str) -> int:
    return len((text or "").split())
