# Signed postings: preserve the zero

RAGLine now has an opt-in fixed-corpus `PostingsRetriever` in the real ask/evaluate/json loop. The unchanged `Retriever` scan remains the default and reference. This is a specialization of established inverted accumulation for this project's signed hashed vectors, not a new retrieval algorithm.

## Real pipeline

```bash
python -m ragline.cli ask --question 'How does TCP provide reliable delivery?' --backend postings --require-grounded
python -m ragline.cli evaluate --backend postings --min-recall 1 --min-mrr .9 --min-grounded 1
python -m ragline.cli json --backend postings
```

`from ragline.postings import PostingsRetriever` accepts the same chunk records and fitted `Embedder` as the original retriever. It copies chunks and fitted embedding state, embeds its fixed corpus once, and builds an immutable numeric index. Access to chunks returns a copy; vectors/postings are tuples. Rebuild after changing documents, dimension or fitted IDF. This is not an online insert/delete/update index. The original mutable retriever is unchanged. Default CLI output is unchanged, including the existing JSON config shape; record the invocation's `--backend` flag when retaining experiment provenance.

The native numeric engine accepts up to 10,000 equal-length vectors of dimension 1..4096, with finite real coordinates of magnitude at most 1,000,000. Booleans are not coordinates. The optional pipeline inherits these bounds; the original scan does not acquire them. Construction validates and retains the dense snapshot, caches original dense norms, and makes one ordered posting per nonzero coordinate. These limits are engineering bounds, not a scale claim.

## What happens to a query

1. Embed using the copied fitted IDF and original SHA-256 signed feature hashing. Nonzero query bins are visited in ascending dimension order.
2. For each such bin, traverse its stored postings and collect each query-coordinate × document-coordinate product for that row. Products stay in original ascending-bin order. Use the running Python interpreter's built-in `sum`, as the original cosine does, rather than a left-fold shortcut. The query norm and cached document norms use the original dense sum/sqrt formula.
3. Assign a score to **every** corpus position. No visited bin means exact zero dot product, not missing data. Shared-bin products can also cancel to zero. Zero norm gets the original zero-cosine convention. Negative scores remain negative and remain eligible.
4. Sort the full binary64 scores descending, stably. Exactly equal scores keep original input order. Only afterward apply the original Python slice for k and round returned hit scores to six decimals. Negative k retains the original slicing behavior; the separate browser replay uses k=0..32.

The original answerer, sentence scoring, extraction, citations, grounding checks and evaluation are unchanged. A sparse candidate-only positive filter would not be equivalent: when matches are negative, an unvisited zero can win. An input-position tie is not a lexical chunk-ID tie. Displayed ties are not necessarily exact-score ties.

## A separate integer-engine witness

The browser is deliberately **not** presented as text/IDF pipeline parity. It takes explicit integer vectors and query coordinates, passes the same numeric engine natively, and recomputes a complete reference report independently in row-first dense JavaScript. It does not reuse the native postings traversal. This separates index correctness from cross-language `log`, Python-version summation and tokenizer behavior.

Input has exactly `schema`, `vectors`, `query`, `k`. Schema is `ragline-signed-postings/v1`; at most 32 rows, 1..64 dimensions, actual JSON integer coordinates in -1000..1000, integer k=0..32, maximum 128 KiB UTF-8. Duplicate/extra/missing fields, nonfinite values, boolean/decimal/exponent coordinates and inconsistent dimensions are rejected. No source text is submitted to a service by the editor. Neither witness nor engine establishes document identity, source authenticity or relevance.

All integer products and squared-norm sums are exact below 2^53. Cosine uses binary64 sqrt/division; ranking uses the unrounded value and input position. JSON exposes exact dots, squared norms, every product, every dimension posting, visited flags, zero kind, complete order and selected positions. Cosine is a six-decimal display, with Python's exact-binary half-even decimal rounding reproduced independently in JavaScript using a BigInt rational conversion. It is not an exact rational cosine certificate. Portability is tested in the stated runtimes, not assumed for every floating implementation.

```bash
python tools/build_postings.py --check
python -m ragline.postings_audit input.json
python -m ragline.postings_audit - < input.json
ragline-postings-audit input.json  # installed console entry point
python -m unittest discover -s tests -v
node --test tests/test_postings_browser.cjs tests/test_postings_chrome.cjs
bash tools/postings_wheel_smoke.sh
```

Valid replay exits 0 and emits one complete JSON report. Invalid/unreadable/oversized input exits 2, writes its error to stderr and emits no report. The audit is not an acceptance or grounding gate. Native query witness access is available as `PostingsRetriever.query_witness(question)`; it exposes raw floating diagnostics separately from the integer replay profile.

## Verification

49 Python tests retain all 31 original tests byte-for-byte. Original retriever, embedder, answerer, corpus and evaluation modules are byte-unchanged. Tests compare the real pipeline across 45 seed/dimension/window configurations and 1,350 queries/k values, including empty/stopword queries and original negative-k slicing. Full fixture evaluations and CLI ask/evaluate/json, both gate outcomes and unchanged stdout are checked against the original scan.

Independent dense arithmetic checks 729 exhaustive signed-cube controls and 400 seeded real-vector matrices. Edge controls cover unvisited zeros ahead of negatives, touched cancellation, zero norms, empty corpus, duplicate rows, exact ties, displayed ties, copied input/metadata and invalid numeric/dimension bounds. The direct row-first arithmetic oracle never calls the index.

24 retained integer reports contain 186 rows and 1,161 product cells. Three Node tests compare every full report and another 200 seeded complete native/browser reports, strict input parsing and binary rounding. One actual-Chrome test checks every rendered row, selected row's full vector/products, all 186 signed-bar positions, containment at 320/390/1280, 48 desktop/phone captures, stale/invalid rejection, trusted text entry and two trusted exports replayed through the native executable. An isolated installed wheel checks nine module origins, all reports via API/module/console, 45 real-pipeline fixture evaluations and both CLI gate sides outside the checkout. CI runs Python 3.10-3.13; latest public/deployed status must be checked at its actual commit before claiming deployment.

## Costs, interpretation and limits

A constructed 32-row, 64-bin control traverses one posting rather than computing 2,048 dense dot-coordinate products. Thirty-one unvisited rows still participate in ranking, above its negative candidate. Those are structural counts on a constructed control, not elapsed time or a general speed ratio. `dense_coordinates` is input size, not total construction operations. Query counters exclude validation, norm work, output creation, ordering, allocation and indexing. Construction keeps the dense snapshot plus the posting structure and cached norms. No lower-memory claim is made. Query still creates all row scores and fully sorts O(N log N); dense queries can traverse every nonzero posting. No WAND/block upper bounds, top-k heap, approximate filter or corpus updates are implemented.

An exact ranking in a signed hashed space can still be semantically wrong. Hash collisions and cancellation are unchanged. This does not make the original lexical representation semantic, nor does it turn extractive copying or token-overlap grounding into entailment, factuality, citation correctness or authentication. Synthetic engineering controls do not establish retrieval-quality gains. Existing eight-question fixture metrics are retained, not a new evaluation or evidence of improved RAG quality.

## Established work and inspected sources

- [Stanford Introduction to Information Retrieval: efficient scoring and ranking](https://www-nlp.stanford.edu/IR-book/html/htmledition/efficient-scoring-and-ranking-1.html): query-dimension postings accumulation and the separate final ranking step. Its positive-weight exposition is not a license to drop zeros/negatives here.
- [scikit-learn HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html) and [feature-hasher source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/feature_extraction/_hash.py): established signed hashing and collision aggregation. RAGLine retains its own original SHA-256 implementation, not scikit-learn's hash.
- [scikit-learn cosine similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html): normalized dot-product definition.
- [Historical scikit-learn collision discussion #7513](https://github.com/scikit-learn/scikit-learn/issues/7513): cancellation examples and design discussion, not a benchmark claim used here.
- [Python built-ins](https://docs.python.org/3/library/functions.html) and [sorting HOWTO](https://docs.python.org/3/howto/sorting.html): runtime summation, rounding and stable reverse sorting. Cross-version bit equality for arbitrary real text-pipeline values is not promised; same-runtime original parity is tested.
- [Quiver sparse-index API](https://docs.rs/quiverdb-query/latest/quiver_query/sparse_index/struct.SparseInvertedIndex.html): another dimension-postings implementation, with a positive-only output contract that is intentionally not adopted here.

No third-party implementation was copied and no trained model, paid API or private corpus is used.
