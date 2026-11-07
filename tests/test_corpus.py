import unittest

from ragline.corpus import build_corpus, chunk_documents


class CorpusTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(build_corpus(seed=1), build_corpus(seed=1))

    def test_seed_changes_fillers(self):
        self.assertNotEqual(build_corpus(seed=1), build_corpus(seed=2))

    def test_size_and_ids_unique(self):
        docs = build_corpus(seed=0, copies=2)
        self.assertEqual(len(docs), 40)
        self.assertEqual(len({d["doc_id"] for d in docs}), 40)

    def test_no_unfilled_slots(self):
        for doc in build_corpus(seed=3):
            self.assertNotIn("{", doc["text"])


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.chunks = chunk_documents(build_corpus(seed=0, copies=2), window=2)

    def test_window_respected(self):
        self.assertTrue(all(len(c["doc_ids"]) <= 2 for c in self.chunks))

    def test_single_topic_per_chunk(self):
        self.assertTrue(all(isinstance(c["topic"], str) for c in self.chunks))

    def test_chunk_ids_unique_and_docs_preserved(self):
        self.assertEqual(len({c["chunk_id"] for c in self.chunks}), len(self.chunks))
        total = sum(len(c["doc_ids"]) for c in self.chunks)
        self.assertEqual(total, 40)


if __name__ == "__main__":
    unittest.main()
