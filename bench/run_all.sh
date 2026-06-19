#!/bin/bash
# Run fixed baseline + multiquery benchmarks.
# Usage: bash bench/run_all.sh   (from repo root)
# LongMemEval dataset must be cloned alongside this repo, or set DATA= manually.
set -e
cd "$(dirname "$0")/.."
DATA="${DATA:-$(dirname "$0")/../../LongMemEval/data}"

echo "=== FIXED BASELINE (4000-char truncation) ==="
python3 bench/longmemeval_retrieval.py --data_dir="$DATA" 2>&1

echo ""
echo "=== MULTIQUERY (gemma3:4b expansion) ==="
python3 bench/longmemeval_multiquery.py --data_dir="$DATA" 2>&1

echo ""
echo "=== DONE ==="
