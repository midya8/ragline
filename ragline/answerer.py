"""Extractive answering with citations, plus grounding verification.

The answerer picks the single best-matching sentence from the retrieved
chunks and answers with it, citing the chunk it came from. The grounding
check then verifies the answer against the cited chunk by token overlap: an
answer sentence that does not share enough content tokens with its cited
source is flagged ungrounded. A corrupted citation is how the check is
tested to fail.
"""

from __future__ import annotations

import re

from .embedder import tokenize

SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")


def split_sentences(text: str) -> list[str]:
    return [m.group(0).strip() for m in SENTENCE_RE.finditer(text)]


def _score_sentence(question_tokens: set[str], sentence: str) -> float:
    tokens = tokenize(sentence)
    if not tokens:
        return 0.0
    overlap = sum(1 for t in tokens if t in question_tokens)
    return overlap / len(tokens)


def answer(question: str, hits: list[dict]) -> dict:
    """Pick the best sentence across hits; cite the chunk it came from."""
    q_tokens = set(tokenize(question))
    best = {"sentence": "", "chunk_id": None, "score": 0.0}
    for hit in hits:
        for sentence in split_sentences(hit["text"]):
            s = _score_sentence(q_tokens, sentence)
            if s > best["score"]:
                best = {"sentence": sentence, "chunk_id": hit["chunk_id"], "score": s}
    if best["chunk_id"] is None:
        return {"question": question, "answer": None, "citations": [], "reason": "no sentence matched"}
    return {
        "question": question,
        "answer": best["sentence"],
        "citations": [best["chunk_id"]],
        "retrieval_scores": {h["chunk_id"]: h["score"] for h in hits},
    }


def grounding_score(answer_text: str, source_text: str) -> float:
    """Fraction of answer content tokens present in the cited source."""
    a_tokens = tokenize(answer_text)
    if not a_tokens:
        return 0.0
    s_tokens = set(tokenize(source_text))
    return sum(1 for t in a_tokens if t in s_tokens) / len(a_tokens)


def check_grounding(result: dict, chunks_by_id: dict[str, dict], min_overlap: float = 0.8) -> dict:
    """Verify each answer sentence against its cited chunk."""
    if result["answer"] is None:
        return {"grounded": False, "score": 0.0, "detail": "no answer produced"}
    chunk = chunks_by_id.get(result["citations"][0])
    if chunk is None:
        return {"grounded": False, "score": 0.0, "detail": f"cited chunk {result['citations'][0]} not in corpus"}
    score = grounding_score(result["answer"], chunk["text"])
    return {
        "grounded": score >= min_overlap,
        "score": round(score, 4),
        "detail": f"answer shares {score:.0%} of content tokens with {result['citations'][0]}",
    }
