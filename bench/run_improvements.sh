#!/bin/bash
# Run both improvement benchmarks and save results.
# Usage: bash bench/run_improvements.sh

set -e
cd "$(dirname "$0")/.."
DATA=/Users/rbr_lpci/Documents/projects/LongMemEval/data

echo "=== TEST 1: Two-pass reranking ==="
python3 bench/longmemeval_rerank.py --split s --data_dir "$DATA" 2>&1
echo ""

echo "=== TEST 2: Multi-query expansion ==="
python3 bench/longmemeval_multiquery.py --split s --data_dir "$DATA" 2>&1
echo ""

echo "=== BOTH DONE ==="
echo "Results:"
ls -la bench/results-rerank-2026-04-15.json bench/results-multiquery-2026-04-15.json
