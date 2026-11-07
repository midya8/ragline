"""RAGLine: retrieval-augmented answering from scratch with verifiable grounding."""

from .answerer import answer, check_grounding, grounding_score
from .corpus import build_corpus, chunk_documents
from .embedder import Embedder, cosine, tokenize
from .evaluate import evaluate
from .retriever import Retriever

__version__ = "0.1.0"

__all__ = [
    "Embedder", "Retriever", "answer", "build_corpus", "check_grounding",
    "chunk_documents", "cosine", "evaluate", "grounding_score", "tokenize",
]
