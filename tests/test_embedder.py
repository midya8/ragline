import unittest

from ragline.embedder import Embedder, cosine, tokenize


class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_strips_stopwords(self):
        self.assertEqual(tokenize("The TCP and the DNS"), ["tcp", "dns"])

    def test_extracts_alphanumerics(self):
        self.assertIn("co2", tokenize("fixes CO2!"))


class EmbedderTests(unittest.TestCase):
    def test_rejects_tiny_dim(self):
        with self.assertRaises(ValueError):
            Embedder(dim=4)

    def test_vector_dim_and_determinism(self):
        e = Embedder(dim=64).fit(["alpha beta", "beta gamma"])
        v1 = e.embed("alpha beta")
        v2 = e.embed("alpha beta")
        self.assertEqual(len(v1), 64)
        self.assertEqual(v1, v2)

    def test_idf_downweights_common_tokens(self):
        e = Embedder(dim=64).fit(["common rare", "common other", "common thing"])
        self.assertLess(e.idf["common"], e.idf["rare"])


class CosineTests(unittest.TestCase):
    def test_identical_vectors(self):
        self.assertAlmostEqual(cosine([1.0, 2.0], [1.0, 2.0]), 1.0)

    def test_orthogonal_and_zero(self):
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertEqual(cosine([0.0, 0.0], [1.0, 1.0]), 0.0)

    def test_related_texts_outrank_unrelated(self):
        e = Embedder(dim=128).fit(["tcp retransmission reliability", "stars hydrogen fusion cores"])
        a = e.embed("tcp retransmission reliability")
        b = e.embed("tcp reliability")
        c = e.embed("stellar fusion")
        self.assertGreater(cosine(a, b), cosine(a, c))


if __name__ == "__main__":
    unittest.main()
