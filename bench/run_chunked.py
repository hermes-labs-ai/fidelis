"""
Run all three benchmark variants in chunks to avoid timeout issues.
Accumulates per-question metrics across chunks and writes final results.
"""
import json
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent
DATA_DIR = Path.home() / "Documents/projects/LongMemEval/data"
TOTAL_QUESTIONS = 470  # LongMemEval S non-abstention
CHUNK = 50

TESTS = [
    {
        "name": "tuned",
        "script": "longmemeval_tuned.py",
        "out": "results-tuned-2026-04-15.json",
    },
    {
        "name": "rerank",
        "script": "longmemeval_rerank.py",
        "out": "results-rerank-2026-04-15.json",
    },
    {
        "name": "multiquery",
        "script": "longmemeval_multiquery.py",
        "out": "results-multiquery-2026-04-15.json",
    },
]


def run_test(test: dict):
    name = test["name"]
    script = BENCH_DIR / test["script"]
    out_file = BENCH_DIR / test["out"]

    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")

    # Run full (the script handles everything internally)
    cmd = [
        sys.executable, str(script),
        "--data_dir", str(DATA_DIR),
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)

    if out_file.exists():
        data = json.load(open(out_file))
        ra = data.get("metrics", {}).get("recall_any", {})
        print(f"\n  {name} DONE: R@1={ra.get('R@1', '?')} R@5={ra.get('R@5', '?')} R@10={ra.get('R@10', '?')}")
    else:
        print(f"\n  {name} FAILED - no output file")


def main():
    for test in TESTS:
        run_test(test)

    print(f"\n{'='*60}")
    print("  ALL TESTS COMPLETE")
    print(f"{'='*60}")

    # Print comparison
    baseline_file = BENCH_DIR / "results-2026-04-15.json"
    if baseline_file.exists():
        bl = json.load(open(baseline_file))
        print(f"\n  Baseline:    R@1={bl['metrics']['recall_any']['R@1']:.1%}  R@5={bl['metrics']['recall_any']['R@5']:.1%}  R@10={bl['metrics']['recall_any']['R@10']:.1%}")

    for test in TESTS:
        f = BENCH_DIR / test["out"]
        if f.exists():
            d = json.load(open(f))
            ra = d["metrics"]["recall_any"]
            print(f"  {test['name']:<12}: R@1={ra['R@1']:.1%}  R@5={ra['R@5']:.1%}  R@10={ra['R@10']:.1%}")


if __name__ == "__main__":
    main()
