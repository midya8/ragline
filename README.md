# ragline

The whole retrieve, answer, cite, check loop of a RAG pipeline in plain Python, without a language model. I built it to understand the plumbing: how retrieval scores come out, where the citation comes from, and how you'd catch an answer that cites the wrong chunk.

```
$ python -m ragline.cli ask --question "How does TCP provide reliable delivery?" --require-grounded
Q: How does TCP provide reliable delivery?
A: TCP provides ordered, reliable delivery using acknowledgements and retransmission.
cited: c25
grounding: answer shares 100% of content tokens with c25 -> GROUNDED
```

How it works:

1. A small corpus (40 documents, 32 chunks) is generated from templates with a fixed seed, so it's the same every run.
2. Chunks become hashed TF-IDF vectors and retrieval is cosine similarity, top-k.
3. The "answer" is the best matching sentence copied out of a retrieved chunk, along with that chunk's id.
4. A grounding check measures how many of the answer's content words appear in the cited chunk. Below the threshold it gets flagged.

```bash
python -m ragline.cli evaluate --min-recall 1.0 --min-mrr 0.9 --min-grounded 1.0
python -m ragline.cli json      # full per-question report
bash examples/demo.sh
```

On the eight built-in questions: recall@3 = 1.00, MRR = 0.94, grounded rate = 1.00. That's a fixture I wrote alongside the code, so it shows the pipeline works, not that it's good at retrieval. The tests also corrupt a citation on purpose to make sure the check catches it.

There's an optional `--backend postings` that does the same ranking through an inverted index instead of scanning every chunk. Rankings come out identical. Notes in POSTINGS.md, and a small browser page at https://midya8.github.io/ragline/.

Limits, plainly: answers are copied sentences, not generated text. The vectors are lexical, so paraphrases won't match. And word overlap with the cited chunk doesn't prove the answer is correct.

Tests: `python -m unittest discover -s tests` (52 tests).
