"""
Real-world Claude Code session retrieval eval.
10 questions written based on topics discussed in recent sessions (Apr 15-17).
For each query, retrieves top-1 session and shows a snippet — user eyeballs correctness.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cogito.recall_sessions import query_sessions


QUESTIONS = [
    {
        "q": "What was the demotion problem we found in the LLM filter?",
        "expect_topic": "verify-guard; filter demotes gold that S1 ranked #1",
    },
    {
        "q": "Why did the learned router fail when we trained it?",
        "expect_topic": "13% positive labels, logistic regression + random forest, 83% CV vs regex 89%",
    },
    {
        "q": "What happened when we tried to inject dates into the chunks for LongMemEval?",
        "expect_topic": "chunk-level date injection, +2pp temporal but -2pp elsewhere, noise floor",
    },
    {
        "q": "What was the R@1 score on the 31-case eval vs LongMemEval hybrid path?",
        "expect_topic": "75% /recall vs 54% /recall_hybrid, 21pt regression",
    },
    {
        "q": "What was the LPCI temporal scaffold experiment that failed?",
        "expect_topic": "Run A, scaffold in filter prompt, -1.2pp overall, -6pp temporal",
    },
    {
        "q": "How did verify-guard work and what did it target?",
        "expect_topic": "restore S1 top-1 when filter demotes it, Pop B ~15 questions",
    },
    {
        "q": "What was the port agent's Gate 6 failure about?",
        "expect_topic": "parity check 0/10, role-structured session vs flat atomic text, rollback",
    },
    {
        "q": "What were the three populations the specialist agent identified in the hardset?",
        "expect_topic": "Pop A 16 both-wrong delta-0 temporal, Pop B 15 filter-demoted, Pop C 1 non-hardset",
    },
    {
        "q": "What was the final R@1 on Run B with flagship on hardset?",
        "expect_topic": "93.4% R@1, 37 flagship calls, $2.33",
    },
    {
        "q": "What's the noise floor we established between baseline runs?",
        "expect_topic": "~1.1-2.1pp noise floor between seed1 and seed2",
    },
]


def run_eval():
    results = []
    for i, q in enumerate(QUESTIONS, 1):
        try:
            res = query_sessions(q["q"], top_k=3)
            top = res[0] if res else None
            def _g(r, k, default=None):
                if r is None:
                    return default
                if isinstance(r, dict):
                    return r.get(k, default)
                return getattr(r, k, default)
            results.append({
                "idx": i,
                "question": q["q"],
                "expected_topic": q["expect_topic"],
                "top_sid": _g(top, "session_id"),
                "top_date": _g(top, "start_ts"),
                "top_score": _g(top, "score"),
                "top_preview": (_g(top, "matched_chunk", "") or "")[:400],
                "alt_2": _g(res[1], "session_id") if len(res) > 1 else None,
                "alt_3": _g(res[2], "session_id") if len(res) > 2 else None,
            })
        except Exception as e:
            results.append({"idx": i, "question": q["q"], "error": str(e)})

    print("=" * 80)
    print(f"Claude Code Session Retrieval Eval — {datetime.now().isoformat(timespec='minutes')}")
    print("=" * 80)
    for r in results:
        print()
        print(f"Q{r['idx']}: {r['question']}")
        if r.get("error"):
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  Expected: {r['expected_topic']}")
        score_str = f"  score={r['top_score']:.3f}" if isinstance(r.get('top_score'), (int, float)) else ""
        print(f"  Top: sid={r['top_sid']}  date={r['top_date']}{score_str}")
        print(f"  Preview: {r['top_preview'][:200]}...")
        print(f"  Alt sids: [{r['alt_2']}, {r['alt_3']}]")

    out_path = str(Path(__file__).parent / "runs" / "claude_code_user_eval.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run_eval()
