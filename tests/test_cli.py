import io
import json
import unittest
from contextlib import redirect_stdout

from ragline.cli import main


def run_cli(args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(args)
    return code, buf.getvalue()


class AskTests(unittest.TestCase):
    def test_grounded_answer_passes_gate(self):
        code, out = run_cli(["ask", "--question", "What does redshift measure?", "--require-grounded"])
        self.assertEqual(code, 0, out)
        self.assertIn("GROUNDED", out)
        self.assertIn("cited:", out)

    def test_unanswerable_fails_require_grounded(self):
        code, _ = run_cli(["ask", "--question", "zqxwv jklhgf mnbvcx?", "--require-grounded"])
        self.assertEqual(code, 1)


class EvaluateTests(unittest.TestCase):
    def test_gates_pass_at_pinned_levels(self):
        code, out = run_cli(["evaluate", "--min-recall", "1.0", "--min-mrr", "0.9", "--min-grounded", "1.0"])
        self.assertEqual(code, 0, out)

    def test_gates_fail_above_achievable(self):
        code, out = run_cli(["evaluate", "--min-recall", "1.0", "--min-mrr", "1.01"])
        self.assertEqual(code, 1)
        self.assertIn("GATE FAILED", out)


class JsonTests(unittest.TestCase):
    def test_report_shape(self):
        code, out = run_cli(["json"])
        self.assertEqual(code, 0)
        r = json.loads(out)
        for key in ("recall_at_k", "mrr", "grounded_rate", "rows", "config"):
            self.assertIn(key, r)


if __name__ == "__main__":
    unittest.main()
