"""Bounded integer coordinate controls, not general floating-order invariance."""
import itertools
import unittest

from ragline.postings import SignedPostingsIndex


class CoordinateTests(unittest.TestCase):
    def test_signed_integer_coordinate_permutations(self):
        vectors = [[1, -2, 0], [0, 1, 3], [0, 0, 0], [-1, 2, 0], [2, 1, -1]]
        queries = [[1, 1, 0], [0, 0, 0], [2, -1, 3]]
        count = 0
        for order in itertools.permutations(range(3)):
            for signs in itertools.product((-1, 1), repeat=3):
                moved = [[row[old] * signs[new] for new, old in enumerate(order)]
                         for row in vectors]
                index = SignedPostingsIndex(moved)
                for query in queries:
                    baseline = SignedPostingsIndex(vectors).score(query, witness=True)
                    transformed = [query[old] * signs[new] for new, old in enumerate(order)]
                    report = index.score(transformed, witness=True)
                    with self.subTest(order=order, signs=signs, query=query):
                        for key in ("scores", "order", "posting_visits", "candidate_count",
                                    "implicit_zero_count", "query_nonzero", "touched", "dots",
                                    "query_norm_squared", "document_norm_squared"):
                            self.assertEqual(report[key], baseline[key])
                        for old_products, new_products in zip(baseline["products"], report["products"]):
                            restored = sorted((order[row["bin"]], row["product"])
                                              for row in new_products)
                            self.assertEqual(restored, [(row["bin"], row["product"])
                                                        for row in old_products])
                    count += 1
        self.assertEqual(count, 144)

    def test_equal_scores_follow_new_input_order(self):
        vectors = [[1, 0], [2, 0], [-1, 0], [0, 0]]
        for order in itertools.permutations(range(4)):
            report = SignedPostingsIndex([vectors[index] for index in order]).score([1, 0])
            restored = [order[index] for index in report["order"]]
            tied = [index for index in order if index in (0, 1)]
            self.assertEqual(restored, tied + [3, 2])

    def test_touched_zero_and_implicit_zero_have_equal_scores_not_equal_evidence(self):
        report = SignedPostingsIndex([[1, -1], [0, 0], [-1, 0]]).score([1, 1], witness=True)
        self.assertEqual(report["scores"][:2], [0, 0])
        self.assertEqual(report["touched"][:2], [True, False])
        self.assertEqual(report["dots"][:2], [0, 0])
        self.assertEqual(report["products"][0], [{"bin": 0, "product": 1},
                                                {"bin": 1, "product": -1}])
        self.assertEqual(report["products"][1], [])
        self.assertEqual(report["order"], [0, 1, 2])
