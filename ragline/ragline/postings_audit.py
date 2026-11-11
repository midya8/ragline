"""Bounded integer-vector replay of the actual signed-postings engine.

This is not a browser reproduction of TF-IDF/logarithms or the text pipeline.
"""
from __future__ import annotations
import argparse
import json
import sys
from .postings import SignedPostingsIndex

SCHEMA = "ragline-signed-postings/v1"


def validate(payload):
    if type(payload) is not dict or set(payload) != {"schema", "vectors", "query", "k"} or payload["schema"] != SCHEMA:
        raise ValueError("expected exact schema, vectors, query and k fields")
    rows, q, k = payload["vectors"], payload["query"], payload["k"]
    if type(q) is not list or not 1 <= len(q) <= 64:
        raise ValueError("query dimension must be 1..64")
    if type(rows) is not list or len(rows) > 32 or any(type(r) is not list or len(r) != len(q) for r in rows):
        raise ValueError("at most 32 vectors of the query dimension")
    if any(type(x) is not int or abs(x) > 1000 for row in [q, *rows] for x in row):
        raise ValueError("all coordinates must be integers in -1000..1000")
    if type(k) is not int or not 0 <= k <= 32:
        raise ValueError("k must be an integer 0..32")
    return rows, q, k


def report(payload):
    vectors, q, k = validate(payload)
    index = SignedPostingsIndex(vectors, dim=len(q))
    witness = index.score(q, witness=True)
    ranks = {i: r+1 for r, i in enumerate(witness["order"])}
    selected = witness["order"][:k]
    def shown(x):
        r = round(x, 6)
        return r if r else 0
    rows = [{"index": i, "vector": list(v), "touched": witness["touched"][i],
             "products": witness["products"][i], "dot": witness["dots"][i],
             "norm_squared": witness["document_norm_squared"][i],
             "cosine": shown(witness["scores"][i]), "rank": ranks[i], "selected": i in selected,
             "zero_kind": ("empty_norm" if witness["query_norm_squared"] == 0 or witness["document_norm_squared"][i] == 0 else "cancellation" if witness["touched"][i] else "implicit") if witness["dots"][i] == 0 else None}
            for i,v in enumerate(vectors)]
    return {"schema": SCHEMA, "input": payload, "query_norm_squared": witness["query_norm_squared"],
            "postings": [{"bin": b, "entries": [{"index": i, "value": value} for i,value in entries]} for b,entries in enumerate(index.postings)],
            "rows": rows, "order": witness["order"], "selected": selected,
            "counts": {key: witness[key] for key in ["query_nonzero", "posting_visits", "candidate_count", "implicit_zero_count", "scan_dot_products", "indexed_dot_products"]},
            "construction": {"dense_coordinates": len(vectors)*len(q), "stored_nonzero": sum(x != 0 for row in vectors for x in row), "cached_norms": len(vectors)},
            "ranking": "descending full binary64 cosine; exact-score ties keep input order; rounding is display-only"}


def _pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError("duplicate JSON field: "+key)
        result[key]=value
    return result


def load(text):
    if len(text.encode("utf-8")) > 131072: raise ValueError("input exceeds 128 KiB")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-finite JSON")))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("input",help="JSON file or - for stdin");a=p.parse_args(argv)
    try:
        if a.input == "-": text=sys.stdin.read(131073)
        else:
            with open(a.input,encoding="utf-8") as f:text=f.read(131073)
        result=report(load(text))
    except (OSError,ValueError,TypeError,RecursionError) as e:
        print("ragline-postings-audit: "+str(e),file=sys.stderr);return 2
    print(json.dumps(result,ensure_ascii=True,separators=(",",":"),allow_nan=False));return 0

if __name__=="__main__":raise SystemExit(main())
