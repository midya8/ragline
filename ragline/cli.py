"""Command-line interface for RAGLine."""

from __future__ import annotations

import argparse
import json
import sys

from .answerer import answer, check_grounding
from .corpus import build_corpus, chunk_documents
from .embedder import Embedder
from .evaluate import evaluate
from .retriever import Retriever
from .postings import PostingsRetriever


def _pipeline(seed: int, copies: int, dim: int, window: int, backend: str = "scan") -> Retriever | PostingsRetriever:
    docs = build_corpus(seed=seed, copies=copies)
    chunks = chunk_documents(docs, window=window)
    embedder = Embedder(dim=dim).fit([c["text"] for c in chunks])
    return (PostingsRetriever if backend == "postings" else Retriever)(chunks, embedder)


def _add_pipeline_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--copies", type=int, default=2)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--backend", choices=("scan", "postings"), default="scan",
                   help="scan is original; postings snapshots the corpus and fitted IDF")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragline", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="answer one question with citations and a grounding check")
    _add_pipeline_args(p_ask)
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("-k", type=int, default=3)
    p_ask.add_argument("--min-overlap", type=float, default=0.8)
    p_ask.add_argument("--require-grounded", action="store_true",
                       help="exit 1 unless the answer passes the grounding check")

    p_eval = sub.add_parser("evaluate", help="fixture evaluation with CI gates")
    _add_pipeline_args(p_eval)
    p_eval.add_argument("-k", type=int, default=3)
    p_eval.add_argument("--min-recall", type=float, default=None)
    p_eval.add_argument("--min-mrr", type=float, default=None)
    p_eval.add_argument("--min-grounded", type=float, default=None)

    p_json = sub.add_parser("json", help="full evaluation report as JSON")
    _add_pipeline_args(p_json)
    p_json.add_argument("-k", type=int, default=3)

    args = parser.parse_args(argv)

    if args.cmd == "ask":
        retriever = _pipeline(args.seed, args.copies, args.dim, args.window, args.backend)
        chunks_by_id = {c["chunk_id"]: c for c in retriever.chunks}
        hits = retriever.query(args.question, k=args.k)
        result = answer(args.question, hits)
        g = check_grounding(result, chunks_by_id, args.min_overlap)
        print(f"Q: {args.question}")
        if result["answer"] is None:
            print(f"A: (no answer - {result['reason']})")
            return 1 if args.require_grounded else 0
        print(f"A: {result['answer']}")
        print(f"cited: {', '.join(result['citations'])}")
        print(f"grounding: {g['detail']} -> {'GROUNDED' if g['grounded'] else 'UNGROUNDED'}")
        if args.require_grounded and not g["grounded"]:
            print("GATE FAILED: answer is not grounded in its citation")
            return 1
        return 0

    if args.cmd == "evaluate":
        retriever = _pipeline(args.seed, args.copies, args.dim, args.window, args.backend)
        rep = evaluate(retriever, k=args.k)
        print(f"recall@{args.k}: {rep['recall_at_k']:.2f} | MRR: {rep['mrr']:.2f} | grounded: {rep['grounded_rate']:.2f} over {rep['questions']} questions")
        failed = []
        if args.min_recall is not None and rep["recall_at_k"] < args.min_recall:
            failed.append(f"recall {rep['recall_at_k']:.2f} < {args.min_recall:.2f}")
        if args.min_mrr is not None and rep["mrr"] < args.min_mrr:
            failed.append(f"MRR {rep['mrr']:.2f} < {args.min_mrr:.2f}")
        if args.min_grounded is not None and rep["grounded_rate"] < args.min_grounded:
            failed.append(f"grounded {rep['grounded_rate']:.2f} < {args.min_grounded:.2f}")
        if failed:
            print("GATE FAILED: " + "; ".join(failed))
            return 1
        return 0

    if args.cmd == "json":
        retriever = _pipeline(args.seed, args.copies, args.dim, args.window, args.backend)
        rep = evaluate(retriever, k=args.k)
        rep["config"] = {"seed": args.seed, "copies": args.copies, "dim": args.dim, "window": args.window}
        print(json.dumps(rep, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
