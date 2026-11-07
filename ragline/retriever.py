"""Top-k cosine retrieval over embedded chunks."""

from __future__ import annotations

from .embedder import Embedder, cosine


class Retriever:
    def __init__(self, chunks: list[dict], embedder: Embedder) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = [embedder.embed(c["text"]) for c in chunks]

    def query(self, question: str, k: int = 3) -> list[dict]:
        qv = self.embedder.embed(question)
        scored = sorted(
            ((cosine(qv, vec), chunk) for chunk, vec in zip(self.chunks, self.vectors)),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            {"score": round(score, 6), "chunk_id": c["chunk_id"], "topic": c["topic"],
             "doc_ids": c["doc_ids"], "text": c["text"]}
            for score, c in scored[:k]
        ]
