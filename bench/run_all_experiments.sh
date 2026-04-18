#!/bin/bash
# Run all 3 experiments sequentially. Just run:
#   bash ~/Documents/projects/cogito-ergo/bench/run_all_experiments.sh
#
# Total time: ~5-10 minutes

cd /Users/rbr_lpci/Documents/projects/cogito-ergo
DATA=/Users/rbr_lpci/Documents/projects/LongMemEval/data

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
