# RAGLine

RAGLine retrieves passages with hashed lexical vectors, copies an answer
sentence and checks token overlap with its cited chunk. It runs offline in
pure Python without a language model or external API.

The default backend scans the corpus. An optional signed-postings backend
computes the same rankings through a fixed-corpus index. Its browser inspector
uses small integer vectors to show the ranking arithmetic, not text retrieval
quality.

## Signed-postings backend

The optional `--backend postings` runs the actual retrieval loop through a fixed-corpus signed-vector index. The original scan remains the default. An unvisited zero can outrank a negative match; touched products can cancel to zero; equal-score ties retain input order. Complete ranking is kept before original slicing and score display.

[Engine contract, reproduction and credited sources](POSTINGS.md) · [Inspectable integer-engine controls](https://midya8.github.io/ragline/)

```bash
python -m ragline.cli ask --question "How does TCP provide reliable delivery?" --backend postings --require-grounded
python -m ragline.cli evaluate --backend postings --min-recall 1 --min-mrr .9 --min-grounded 1
python -m unittest discover -s tests -v
node --test tests/test_postings_browser.cjs tests/test_postings_chrome.cjs
bash tools/postings_wheel_smoke.sh
```

52 Python tests retain all 31 originals unchanged. Bounded integer coordinate controls add 144 signed-permutation comparisons, 24 tied-row reorderings and a cancellation-versus-implicit-zero witness. Coordinate permutations can change floating-point accumulation, so the integer checks do not assert general floating-order invariance. Independent references cover 729 exhaustive signed-cube controls, 400 seeded real-vector comparisons and 1,350 real hashed-pipeline query/k comparisons. Four Node/actual-Chrome tests check 24 complete reports, another 200 seeded native/browser reports, all 186 rows / 1,161 product cells / 186 SVG marks, responsive views and two trusted exports replayed natively. An isolated installed wheel checks API/module/console reports, 45 pipeline fixture comparisons, both gate sides and nine module origins.

The browser is explicitly a bounded **integer-vector engine** reference, not text/IDF parity or a RAG-quality benchmark. Construction retains dense vectors plus postings and norms; query still creates all scores and sorts the full corpus. One constructed 32×64 control visits one posting instead of 2,048 dense dot-coordinate products, but that is not a timing, memory or general speed claim. Signed hashing, extractive answers and lexical grounding retain their original limits. The snapshot must be rebuilt after corpus/fit changes. No new retrieval-method or semantic-quality claim is made.

The original project and eight-question fixture below are retained. These are engineering checks, not evidence of better retrieval quality.

---

Retrieval-augmented answering built from scratch in pure Python: deterministic sparse embeddings, top-k retrieval, extractive answers with citations, and a lexical grounding check that scores token overlap between each answer and the chunk it cites - fully offline, no models, no APIs.

RAGLine runs the whole RAG loop as inspectable steps:

- **corpus** - a deterministic document set generated from fixed templates (one seed reproduces it bit for bit), chunked into paragraph windows with stable ids.
- **embed** - token-hashed sparse TF vectors with IDF weighting computed over the chunk set; cosine similarity is the retrieval score.
- **retrieve** - top-k chunks per question with descending scores.
- **answer** - the best-matching sentence from the retrieved chunks, returned with the id of the chunk it came from. Every answer carries a citation.
- **verify** - a lexical grounding check scores the fraction of answer content tokens present in the cited chunk; below the threshold the answer is flagged ungrounded and `--require-grounded` fails the run. In the suite's negative control, corrupting the citation to an unrelated chunk makes the check flag the answer.

## Eight-question fixture

Default fixture (seed 0, 40 documents, 32 chunks, dim 256, window 2), from `ragline evaluate`:

| metric | value |
|---|---|
| recall@3 (correct topic in top 3) | 1.00 over 8 questions |
| MRR | 0.94 |
| grounded-answer rate | 1.00 |

CI pins these with `--min-recall 1.0 --min-mrr 0.9 --min-grounded 1.0`, and a negative control corrupts a citation to an unrelated chunk so the lexical check flags it. These eight questions are development fixtures, not a held-out retrieval benchmark.

## What is verified

- **Determinism end to end** - same seed, same corpus, same rankings, same answers; pinned in tests.
- **Fixture retrieval** - eight questions with known topics gate recall@k and MRR.
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
- `tests/` - the original 31 unit tests cover corpus determinism, embeddings, retrieval ranking, answering, both grounding directions, evaluation metrics, and every CLI gate.

## Limits

- Answers are extractive: a sentence copied from the corpus, never generated text. There is no language model here, by design - the project demonstrates the retrieval-and-verification loop, not fluent generation.
- The embedding is lexical (hashed TF-IDF-style), not a semantic representation. Shared hash bins can produce similarity without shared words; paraphrase retrieval quality is not evaluated.
- The grounding check measures lexical overlap between answer and cited chunk. It flags the tested wrong-citation controls, but does not prove entailment, factuality or correct interpretation.
