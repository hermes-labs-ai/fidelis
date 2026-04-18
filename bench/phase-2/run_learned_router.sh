#!/bin/bash
# Run Step 4: embed queries + train learned router
# Usage: bash bench/phase-2/run_learned_router.sh
cd /Users/rbr_lpci/Documents/projects/cogito-ergo
curl -s http://localhost:11434/api/embed -d '{"model":"nomic-embed-text","input":["warm"]}' > /dev/null
python3 -u bench/phase-2/learned_router.py
