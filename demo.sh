#!/bin/bash
# End-to-end demo: build the venv if needed, run the evaluation
# (BM25 vs TF-IDF over the synthetic corpus), then run unit tests.
# All metrics printed are measured from this run. No external APIs.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "== Creating venv and installing dependencies =="
  python3 -m venv --system-site-packages .venv
  .venv/bin/pip install -q -r requirements.txt
fi

echo "== 1. Run evaluation (BM25 vs TF-IDF) =="
.venv/bin/python evaluate.py

echo ""
echo "== 2. Unit tests =="
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -3
echo ""
echo "Demo complete. Full per-question results: results/report.json"
