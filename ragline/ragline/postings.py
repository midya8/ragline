"""Fixed-corpus signed sparse accumulation. No score filtering or approximate ranking."""
from __future__ import annotations

import copy
import math
from numbers import Real

from .embedder import Embedder


def _vector(value, dim=None):
    values = tuple(value)
    if dim is not None and len(values) != dim:
        raise ValueError("query dimension differs from index")
    if not values or len(values) > 4096:
        raise ValueError("vector dimension must be 1..4096")
    if any(isinstance(x, bool) or not isinstance(x, Real) or abs(x) > 1e6 or not math.isfinite(x) for x in values):
        raise ValueError("vector values must be finite real numbers with magnitude <= 1000000")
    return values


class SignedPostingsIndex:
    """Immutable numeric snapshot; all documents, including zeros/negatives, rank.

    Construction scans and retains the dense numeric snapshot and caches its norms.
    Query saves dot products, not necessarily time or memory. Sorting is still full.
    The numeric bounds avoid non-finite intermediate products. Sum uses the running
    interpreter's built-in, just as the original cosine implementation does.
    """

    def __init__(self, vectors, *, dim=None):
        if dim is not None and (type(dim) is not int or not 1 <= dim <= 4096):
            raise ValueError("dimension must be an integer 1..4096")
        vectors = tuple(vectors)
        if len(vectors) > 10000:
            raise ValueError("at most 10000 vectors")
        if not vectors:
            if type(dim) is not int or not 1 <= dim <= 4096:
                raise ValueError("empty index needs dimension 1..4096")
            self._dim = dim
        else:
            first = _vector(vectors[0])
            self._dim = len(first)
            if dim is not None and dim != self._dim:
                raise ValueError("dimension differs from vectors")
        self._vectors = (first,) + tuple(_vector(v, self._dim) for v in vectors[1:]) if vectors else ()
        self._norms = tuple(math.sqrt(sum(x*x for x in v)) for v in self._vectors)
        self._postings = tuple(tuple((i, v[b]) for i, v in enumerate(self._vectors) if v[b] != 0) for b in range(self._dim))

    @property
    def vectors(self):
        return self._vectors

    @property
    def postings(self):
        return self._postings

    @property
    def dim(self):
        return self._dim

    def score(self, query, *, witness=False):
        q = _vector(query, self._dim)
        qnorm = math.sqrt(sum(x*x for x in q))
        products = {}
        visits = 0
        for b, x in enumerate(q):
            if x != 0:
                for i, y in self._postings[b]:
                    products.setdefault(i, []).append((b, x*y))
                    visits += 1
        # Ascending-bin order matches original dense cosine sum. Omitted zeros
        # do not change sum; no incremental += shortcut across Python versions.
        dots = [sum(product for _, product in products.get(i, ())) for i in range(len(self._vectors))]
        scores = [dot/(qnorm*norm) if qnorm and norm else 0.0 for dot, norm in zip(dots, self._norms)]
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        result = {"scores": scores, "order": order, "posting_visits": visits,
                  "query_nonzero": sum(x != 0 for x in q), "candidate_count": len(products),
                  "implicit_zero_count": len(scores)-len(products),
                  "scan_dot_products": self._dim*len(scores),
                  "indexed_dot_products": visits}
        if witness:
            result.update(query_norm_squared=sum(x*x for x in q),
                          document_norm_squared=[sum(x*x for x in v) for v in self._vectors],
                          dots=dots,
                          products=[[{"bin": b, "product": v} for b, v in products.get(i, ())] for i in range(len(scores))],
                          touched=[i in products for i in range(len(scores))])
        return result


class PostingsRetriever:
    """Opt-in immutable fixed-corpus snapshot with original Retriever hit shape.

    Rebuild to change documents, vectors, dimension or fitted IDF. Public metadata
    access returns copies; source caller mutations cannot silently stale postings.
    """

    def __init__(self, chunks, embedder: Embedder):
        self._chunks = copy.deepcopy(list(chunks))
        self._embedder = copy.deepcopy(embedder)
        self._index = SignedPostingsIndex([self._embedder.embed(c["text"]) for c in self._chunks], dim=self._embedder.dim)

    @property
    def chunks(self):
        return copy.deepcopy(self._chunks)

    @property
    def vectors(self):
        return self._index.vectors

    def query(self, question, k=3):
        result = self._index.score(self._embedder.embed(question))
        return [{"score": round(result["scores"][i], 6), "chunk_id": c["chunk_id"],
                 "topic": c["topic"], "doc_ids": copy.deepcopy(c["doc_ids"]), "text": c["text"]}
                for i in result["order"][:k] for c in [self._chunks[i]]]

    def query_witness(self, question):
        return self._index.score(self._embedder.embed(question), witness=True)
