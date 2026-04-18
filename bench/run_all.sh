#!/bin/bash
# Run fixed baseline + multiquery benchmarks.
# Usage: bash ~/Documents/projects/cogito-ergo/bench/run_all.sh
set -e
cd /Users/rbr_lpci/Documents/projects/cogito-ergo
DATA=/Users/rbr_lpci/Documents/projects/LongMemEval/data

echo "=== FIXED BASELINE (4000-char truncation) ==="
python3 bench/longmemeval_retrieval.py --data_dir="$DATA" 2>&1

echo ""
echo "=== MULTIQUERY (gemma3:4b expansion) ==="
python3 bench/longmemeval_multiquery.py --data_dir="$DATA" 2>&1

echo ""
echo "=== DONE ==="
