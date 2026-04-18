#!/usr/bin/env python3
"""
Regrade QA eval results using LLM grading for nuanced answers.

Problem: QA eval produces correct answers (e.g., "Business Administration") but
grader marks them wrong because QA model wraps them in conversation format
(e.g., "[user]: I graduated with a degree in Business Administration").

Strategy:
1. Read qa_eval_v2_*.json results file
2. For each entry:
   a. Try exact containment: gold appears in qa_answer (case-insensitive)
   b. If not, use qwen-turbo via API: "Does this answer match? Gold: X, Given: Y. Reply YES or NO only."
3. Write updated results with corrected `correct` field
4. Print summary: original accuracy vs regraded accuracy, per category

Usage:
    DASHSCOPE_API_KEY=sk-... python3 regrade_qa.py --input qa_eval_v2_runP-v35_qwen-max_top5.json
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path


DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
FILTER_MODEL = "qwen-turbo"
API_TIMEOUT = 10


def exact_match(gold: str, qa_answer: str) -> bool:
    """Check if gold answer is contained in qa_answer (case-insensitive)."""
    gold_lower = gold.lower().strip()
    qa_lower = qa_answer.lower()
    return gold_lower in qa_lower


def llm_grade(gold: str, qa_answer: str) -> bool | None:
    """
    Use qwen-turbo to grade if the answer matches the gold standard.
    Returns True if match, False if no match, None if API fails.
    """
    if not DASHSCOPE_API_KEY:
        return None

    # Truncate long answers to avoid token explosion
    qa_snippet = qa_answer[:500] if len(qa_answer) > 500 else qa_answer
    gold_snippet = gold[:200] if len(gold) > 200 else gold

    prompt = f"""Does this answer match the gold standard?
Gold: {gold_snippet}
Given: {qa_snippet}
Reply YES or NO only."""

    body = json.dumps({
        "model": FILTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 10,
    })

    try:
        req = urllib.request.Request(
            DASHSCOPE_URL,
            data=body.encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            data = json.loads(resp.read())
        raw = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
        return raw.startswith("YES")
    except Exception as e:
        print(f"  [grade API error] {e}", file=sys.stderr)
        return None


def regrade_entry(entry: dict) -> tuple[bool, str]:
    """
    Regrade a single QA entry.
    Returns (new_correct, method) where method is "exact", "llm", or "fallback".
    """
    gold = entry.get("gold_answer", "")
    qa_answer = entry.get("qa_answer", "")

    # Try exact containment first
    if exact_match(gold, qa_answer):
        return (True, "exact")

    # Try LLM grading
    llm_result = llm_grade(gold, qa_answer)
    if llm_result is not None:
        return (llm_result, "llm")

    # Fallback: keep original grading
    return (entry.get("qa_correct", False), "fallback")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input qa_eval_v2_*.json file")
    parser.add_argument("--output", help="Output file (default: input with _regraded suffix)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load results
    print(f"Loading {input_path}...", file=sys.stderr)
    with open(input_path) as f:
        results = json.load(f)

    # Regrade each entry
    stats = {
        "total": len(results),
        "original_correct": 0,
        "regraded_correct": 0,
        "improved": 0,
        "degraded": 0,
        "methods": defaultdict(int),
        "by_category": defaultdict(lambda: {
            "total": 0,
            "orig_correct": 0,
            "new_correct": 0,
        }),
    }

    print(f"Regrading {len(results)} entries...", file=sys.stderr)
    for i, entry in enumerate(results):
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(results)}", file=sys.stderr)
            time.sleep(0.1)  # Rate limit

        orig_correct = entry.get("qa_correct", False)
        new_correct, method = regrade_entry(entry)

        # Update entry
        entry["qa_correct_original"] = orig_correct
        entry["qa_correct"] = new_correct
        entry["regrade_method"] = method

        # Update stats
        if orig_correct:
            stats["original_correct"] += 1
        if new_correct:
            stats["regraded_correct"] += 1
        if new_correct and not orig_correct:
            stats["improved"] += 1
        if not new_correct and orig_correct:
            stats["degraded"] += 1
        stats["methods"][method] += 1

        # By category
        qtype = entry.get("qtype", "unknown")
        stats["by_category"][qtype]["total"] += 1
        if orig_correct:
            stats["by_category"][qtype]["orig_correct"] += 1
        if new_correct:
            stats["by_category"][qtype]["new_correct"] += 1

    # Write output
    output_path = args.output
    if not output_path:
        stem = input_path.stem + "_regraded"
        output_path = input_path.parent / (stem + ".json")
    else:
        output_path = Path(output_path)

    print(f"Writing {output_path}...", file=sys.stderr)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70, file=sys.stderr)
    print("REGRADE SUMMARY", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Total entries: {stats['total']}", file=sys.stderr)
    orig_acc = 100 * stats["original_correct"] / stats["total"]
    new_acc = 100 * stats["regraded_correct"] / stats["total"]
    print(f"Original accuracy: {stats['original_correct']}/{stats['total']} ({orig_acc:.1f}%)", file=sys.stderr)
    print(f"Regraded accuracy: {stats['regraded_correct']}/{stats['total']} ({new_acc:.1f}%)", file=sys.stderr)
    print(f"Delta: {new_acc - orig_acc:+.1f}%", file=sys.stderr)
    print(f"Improved: {stats['improved']}, Degraded: {stats['degraded']}", file=sys.stderr)

    print("\nGrading methods:", file=sys.stderr)
    for method, count in sorted(stats["methods"].items()):
        pct = 100 * count / stats["total"]
        print(f"  {method}: {count} ({pct:.1f}%)", file=sys.stderr)

    print("\nBy category:", file=sys.stderr)
    for qtype in sorted(stats["by_category"].keys()):
        cat = stats["by_category"][qtype]
        orig_acc = 100 * cat["orig_correct"] / cat["total"] if cat["total"] > 0 else 0
        new_acc = 100 * cat["new_correct"] / cat["total"] if cat["total"] > 0 else 0
        print(f"  {qtype}:", file=sys.stderr)
        print(f"    total: {cat['total']}", file=sys.stderr)
        print(f"    original: {orig_acc:.1f}% ({cat['orig_correct']}/{cat['total']})", file=sys.stderr)
        print(f"    regraded: {new_acc:.1f}% ({cat['new_correct']}/{cat['total']})", file=sys.stderr)

    print(f"\nOutput: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
