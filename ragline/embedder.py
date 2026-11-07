"""Deterministic sparse embedding: token-hashed TF vectors with IDF weighting.

No trained model: each token hashes into one of `dim` bins (with a sign flip
from a second hash bit), vectors accumulate term frequency, and an IDF factor
computed over the chunk set downweights ubiquitous tokens. Cosine similarity
over these vectors is the retrieval score.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "to", "and", "is", "are", "by", "for",
    "with", "from", "into", "on", "using", "they", "them", "it", "its",
})


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def _bucket(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    return idx, sign


class Embedder:
    def __init__(self, dim: int = 256) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim
        self.idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> "Embedder":
        df: Counter[str] = Counter()
        for text in texts:
            for token in set(tokenize(text)):
                df[token] += 1
        n = max(len(texts), 1)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        return self

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token, count in Counter(tokenize(text)).items():
            idx, sign = _bucket(token, self.dim)
            vec[idx] += sign * count * self.idf.get(token, 1.0)
        return vec


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
