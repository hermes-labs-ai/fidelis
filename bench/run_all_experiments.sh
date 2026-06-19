#!/bin/bash
# Run all 3 experiments sequentially. Just run:
#   bash bench/run_all_experiments.sh   (from repo root)
# LongMemEval dataset must be cloned alongside this repo, or set DATA= manually.
#
# Total time: ~5-10 minutes

cd "$(dirname "$0")/.."
DATA="${DATA:-$(dirname "$0")/../../LongMemEval/data}"

echo "=========================================="
echo "  TASK 1: Nomic prefix fix"
echo "=========================================="
curl -s http://localhost:11434/api/embed -d '{"model":"nomic-embed-text","input":["warm"]}' > /dev/null
python3 -u bench/longmemeval_prefix.py --data_dir="$DATA" 2>&1

echo ""
echo "=========================================="
echo "  TASK 2: BM25 + Dense Hybrid"
echo "=========================================="
curl -s http://localhost:11434/api/embed -d '{"model":"nomic-embed-text","input":["warm"]}' > /dev/null
python3 -u bench/longmemeval_hybrid.py --data_dir="$DATA" 2>&1

echo ""
echo "=========================================="
echo "  TASK 3: Turn-Level Chunking"
echo "=========================================="
curl -s http://localhost:11434/api/embed -d '{"model":"nomic-embed-text","input":["warm"]}' > /dev/null
python3 -u bench/longmemeval_turnlevel.py --data_dir="$DATA" 2>&1

echo ""
echo "=========================================="
echo "  ALL EXPERIMENTS DONE"
echo "=========================================="
echo "Results files:"
ls -la bench/results-prefix-2026-04-15.json bench/results-hybrid-2026-04-15.json bench/results-turnlevel-2026-04-15.json 2>/dev/null
