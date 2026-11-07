"""Fixture-driven evaluation: retrieval recall@k, MRR, and grounded-answer rate."""

from __future__ import annotations

from .answerer import answer, check_grounding
from .retriever import Retriever

QA_FIXTURES = [
    {"question": "Where do the light-dependent reactions occur?", "topic": "photosynthesis"},
    {"question": "What fixes carbon dioxide in the Calvin cycle?", "topic": "photosynthesis"},
    {"question": "How does TCP provide reliable delivery?", "topic": "networking"},
    {"question": "What does DNS resolve hostnames into?", "topic": "networking"},
    {"question": "What does a B-tree index keep sorted?", "topic": "databases"},
    {"question": "What does write-ahead logging record?", "topic": "databases"},
    {"question": "What does redshift measure?", "topic": "astronomy"},
    {"question": "How are exoplanets detected?", "topic": "astronomy"},
]


def evaluate(retriever: Retriever, k: int = 3, min_overlap: float = 0.8) -> dict:
    chunks_by_id = {c["chunk_id"]: c for c in retriever.chunks}
    recall_hits = 0
    rr_sum = 0.0
    grounded = 0
    rows = []
    for fx in QA_FIXTURES:
        hits = retriever.query(fx["question"], k=k)
        rank = next((i + 1 for i, h in enumerate(hits) if h["topic"] == fx["topic"]), None)
        recall_hits += rank is not None
        rr_sum += 1.0 / rank if rank else 0.0
        result = answer(fx["question"], hits)
        g = check_grounding(result, chunks_by_id, min_overlap)
        grounded += g["grounded"]
        rows.append({
            "question": fx["question"],
            "expected_topic": fx["topic"],
            "rank": rank,
            "answer": result["answer"],
            "citations": result["citations"],
            "grounding": g,
        })
    n = len(QA_FIXTURES)
    return {
        "k": k,
        "questions": n,
        "recall_at_k": round(recall_hits / n, 4),
        "mrr": round(rr_sum / n, 4),
        "grounded_rate": round(grounded / n, 4),
        "rows": rows,
    }
