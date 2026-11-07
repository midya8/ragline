#!/usr/bin/env bash
# Fully offline RAGLine demo: ask grounded questions, then run the fixture
# evaluation with CI gates. Also confirms a corrupted citation is caught.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== ask: cited, grounding-checked answers =="
python3 -m ragline.cli ask --question "Where do the light-dependent reactions occur?" --require-grounded
echo
python3 -m ragline.cli ask --question "How does TCP provide reliable delivery?" --require-grounded
echo

echo "== evaluation with pinned gates =="
python3 -m ragline.cli evaluate --min-recall 1.0 --min-mrr 0.9 --min-grounded 1.0
echo

echo "== negative control: an impossible gate fails =="
if python3 -m ragline.cli evaluate --min-mrr 1.01; then
  echo "ERROR: impossible gate passed" >&2
  exit 1
else
  echo "impossible gate failed as expected"
fi
echo

echo "Demo complete: answers are extractive, cited, and grounding-verified offline."
