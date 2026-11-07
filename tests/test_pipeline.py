import unittest

from ragline.answerer import answer, check_grounding, grounding_score, split_sentences
from ragline.corpus import build_corpus, chunk_documents
from ragline.embedder import Embedder
from ragline.evaluate import evaluate
from ragline.retriever import Retriever


def make_retriever(seed=0):
    docs = build_corpus(seed=seed, copies=2)
    chunks = chunk_documents(docs, window=2)
    emb = Embedder(dim=256).fit([c["text"] for c in chunks])
    return Retriever(chunks, emb), chunks


class RetrieverTests(unittest.TestCase):
    def test_top_hit_correct_topic(self):
        r, _ = make_retriever()
        hits = r.query("Where do the light-dependent reactions occur?", k=1)
        self.assertEqual(hits[0]["topic"], "photosynthesis")

    def test_scores_descending_and_k(self):
        r, _ = make_retriever()
        hits = r.query("What does DNS resolve?", k=3)
        self.assertEqual(len(hits), 3)
        scores = [h["score"] for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_deterministic_ranking(self):
        r1, _ = make_retriever()
        r2, _ = make_retriever()
        q = "How does TCP provide reliable delivery?"
        self.assertEqual([h["chunk_id"] for h in r1.query(q)], [h["chunk_id"] for h in r2.query(q)])


class AnswererTests(unittest.TestCase):
    def test_splits_sentences(self):
        self.assertEqual(split_sentences("One. Two! Three?"), ["One.", "Two!", "Three?"])

    def test_answer_cites_real_chunk(self):
        r, chunks = make_retriever()
        res = answer("What does redshift measure?", r.query("What does redshift measure?", k=3))
        ids = {c["chunk_id"] for c in chunks}
        self.assertIn(res["citations"][0], ids)
        self.assertIsNotNone(res["answer"])

    def test_unanswerable_returns_none(self):
        r, _ = make_retriever()
        res = answer("zqxwv jklhgf mnbvcx?", r.query("zqxwv jklhgf mnbvcx?", k=3))
        self.assertIsNone(res["answer"])
        self.assertEqual(res["citations"], [])


class GroundingTests(unittest.TestCase):
    def test_score_bounds(self):
        self.assertEqual(grounding_score("tcp retransmission", "tcp uses retransmission."), 1.0)
        self.assertEqual(grounding_score("quantum zebra", "tcp uses retransmission."), 0.0)

    def test_correct_citation_grounded(self):
        r, chunks = make_retriever()
        by_id = {c["chunk_id"]: c for c in chunks}
        res = answer("What does redshift measure?", r.query("What does redshift measure?", k=3))
        self.assertTrue(check_grounding(res, by_id)["grounded"])

    def test_corrupted_citation_flagged(self):
        r, chunks = make_retriever()
        by_id = {c["chunk_id"]: c for c in chunks}
        res = answer("What does redshift measure?", r.query("What does redshift measure?", k=3))
        wrong = next(c["chunk_id"] for c in chunks if c["topic"] == "databases")
        res["citations"] = [wrong]
        g = check_grounding(res, by_id)
        self.assertFalse(g["grounded"])

    def test_missing_chunk_flagged(self):
        g = check_grounding({"answer": "anything.", "citations": ["nope"]}, {})
        self.assertFalse(g["grounded"])


class EvaluateTests(unittest.TestCase):
    def test_fixture_metrics(self):
        r, _ = make_retriever()
        rep = evaluate(r, k=3)
        self.assertEqual(rep["recall_at_k"], 1.0)
        self.assertGreaterEqual(rep["mrr"], 0.9)
        self.assertEqual(rep["grounded_rate"], 1.0)
        self.assertEqual(len(rep["rows"]), 8)


if __name__ == "__main__":
    unittest.main()
