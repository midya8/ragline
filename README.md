# RAGLine

Retrieval-augmented answering built from scratch in pure Python: deterministic sparse embeddings, top-k retrieval, extractive answers with citations, and a lexical grounding check that scores token overlap between each answer and the chunk it cites - fully offline, no models, no APIs.

RAGLine runs the whole RAG loop as inspectable steps:

- **corpus** - a deterministic document set generated from fixed templates (one seed reproduces it bit for bit), chunked into paragraph windows with stable ids.
- **embed** - token-hashed sparse TF vectors with IDF weighting computed over the chunk set; cosine similarity is the retrieval score.
- **retrieve** - top-k chunks per question with descending scores.
- **answer** - the best-matching sentence from the retrieved chunks, returned with the id of the chunk it came from. Every answer carries a citation.
- **verify** - a lexical grounding check scores the fraction of answer content tokens present in the cited chunk; below the threshold the answer is flagged ungrounded and `--require-grounded` fails the run. In the suite's negative control, corrupting the citation to an unrelated chunk makes the check flag the answer.

## Measured results

Default fixture (seed 0, 40 documents, 32 chunks, dim 256, window 2), from `ragline evaluate`:

| metric | value |
|---|---|
| recall@3 (correct topic in top 3) | 1.00 over 8 questions |
| MRR | 0.94 |
| grounded-answer rate | 1.00 |

CI pins these with `--min-recall 1.0 --min-mrr 0.9 --min-grounded 1.0`, and the test suite proves the check has teeth: corrupting a citation to point at an unrelated chunk makes the grounding check flag the answer.

## What is verified

- **Determinism end to end** - same seed, same corpus, same rankings, same answers; pinned in tests.
- **Retrieval quality** - fixture questions with known topics gate recall@k and MRR.
- **Grounding, both directions, within scope** - on the tested fixture, correct citations pass at 100% token overlap, and the tested negative controls (a citation corrupted to an unrelated chunk, a missing chunk) are flagged. The check is lexical: it detects these tested wrong-citation cases, not arbitrary misinterpretation.

## Usage

```bash
python3 -m ragline.cli ask --question "How does TCP provide reliable delivery?" --require-grounded
python3 -m ragline.cli evaluate --min-recall 1.0 --min-mrr 0.9 --min-grounded 1.0
python3 -m ragline.cli json          # full report with per-question rows
bash examples/demo.sh                # offline demo, both gate sides
```

## Layout

- `ragline/corpus.py` - deterministic templated corpus and chunking.
- `ragline/embedder.py` - token-hashed TF-IDF-style sparse vectors and cosine.
- `ragline/retriever.py` - top-k retrieval.
- `ragline/answerer.py` - extractive answers, citations, grounding check.
- `ragline/evaluate.py` - fixture evaluation (recall@k, MRR, grounded rate).
- `tests/` - 31 unit tests covering corpus determinism, embeddings, retrieval ranking, answering, both grounding directions, evaluation metrics, and every CLI gate.

## Limits

- Answers are extractive: a sentence copied from the corpus, never generated text. There is no language model here, by design - the project demonstrates the retrieval-and-verification loop, not fluent generation.
- The embedding is lexical (hashed TF-IDF-style); paraphrases that share no content tokens will not match. Semantic embeddings would need a trained model, which the no-dependency rule excludes.
- The grounding check is lexical overlap between answer and cited chunk. It catches the tested wrong-citation cases and unsupported wording, not subtle misinterpretation of a correctly cited source.
