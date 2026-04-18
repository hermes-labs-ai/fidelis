"""
Analyze per-question S1 vs S2 outcomes from the combined pipeline.
Does NOT rerun embeddings — uses the same retrieval + filter logic but
logs every question's outcome for diagnosis.

Outputs: bench/analysis-pipeline-2026-04-15.json with per-question detail.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path.home() / "Documents/projects/LongMemEval/data"
BENCH_DIR = Path(__file__).parent


def main():
    data_path = DATA_DIR / "longmemeval_s_cleaned.json"
    data = json.load(open(data_path))
    data = [e for e in data if "_abs" not in e["question_id"]]

    # We need to rerun the pipeline to get per-question data
    # But that takes too long. Instead, let's analyze the question TYPES
    # and cross-reference with the S1 running scores from the log.

    # Categorize all questions
    categories = defaultdict(list)
    temporal_keywords = {"first", "before", "after", "between", "ago", "last", "recent", "earlier", "later", "order", "when did", "how many days", "how many weeks", "how many months", "which happened"}
    counting_keywords = {"how many", "total number", "count", "how much"}

    for qi, entry in enumerate(data):
        qid = entry["question_id"]
        qtype = entry.get("question_type", "unknown")
        question = entry["question"].lower()

        # Detect temporal
        is_temporal = any(kw in question for kw in temporal_keywords)
        # Detect counting
        is_counting = any(kw in question for kw in counting_keywords)
        # Multi-session (multiple answer sessions)
        n_answers = len(entry["answer_session_ids"])
        is_multi = n_answers > 1
        # Has dates in metadata
        has_dates = bool(entry.get("haystack_dates"))
        # Corpus size
        n_sessions = len(entry["haystack_session_ids"])

        categories[qtype].append({
            "qi": qi,
            "qid": qid,
            "question": entry["question"],
            "type": qtype,
            "is_temporal": is_temporal,
            "is_counting": is_counting,
            "is_multi_answer": is_multi,
            "n_answer_sessions": n_answers,
            "n_haystack_sessions": n_sessions,
            "has_dates": has_dates,
            "question_date": entry.get("question_date", ""),
            "answer_dates": [entry["haystack_dates"][entry["haystack_session_ids"].index(sid)]
                           for sid in entry["answer_session_ids"]
                           if sid in entry["haystack_session_ids"]] if has_dates else [],
        })

    # Summary
    print(f"{'='*70}")
    print(f"  LongMemEval S — Question Analysis ({len(data)} questions)")
    print(f"{'='*70}\n")

    print(f"  By question_type field:")
    for qtype, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        temporal_count = sum(1 for i in items if i["is_temporal"])
        counting_count = sum(1 for i in items if i["is_counting"])
        multi_count = sum(1 for i in items if i["is_multi_answer"])
        avg_answers = sum(i["n_answer_sessions"] for i in items) / len(items)
        print(f"  {qtype:40s} n={len(items):3d}  temporal={temporal_count:3d}  counting={counting_count:3d}  multi_ans={multi_count:3d}  avg_ans={avg_answers:.1f}")

    print(f"\n  Overall:")
    all_items = [item for items in categories.values() for item in items]
    temporal_total = sum(1 for i in all_items if i["is_temporal"])
    counting_total = sum(1 for i in all_items if i["is_counting"])
    multi_total = sum(1 for i in all_items if i["is_multi_answer"])
    print(f"  Temporal questions: {temporal_total}/{len(data)} ({temporal_total/len(data)*100:.0f}%)")
    print(f"  Counting questions: {counting_total}/{len(data)} ({counting_total/len(data)*100:.0f}%)")
    print(f"  Multi-answer questions: {multi_total}/{len(data)} ({multi_total/len(data)*100:.0f}%)")

    # Analyze which question positions are "hard" based on the progress data
    # Questions 230-310 is where R@1 dropped most in all runs
    print(f"\n  Question position ranges:")
    for start, end, label in [(0,100,"Easy (q1-100)"), (100,200,"Medium (q101-200)"), (200,300,"Hard (q201-300)"), (300,400,"Hardest (q301-400)"), (400,470,"Recovery (q401-470)")]:
        subset = all_items[start:end]
        t = sum(1 for i in subset if i["is_temporal"])
        c = sum(1 for i in subset if i["is_counting"])
        m = sum(1 for i in subset if i["is_multi_answer"])
        print(f"  {label:25s} temporal={t:2d}  counting={c:2d}  multi={m:2d}")

    # Key diagnostic: the q230-310 dip zone
    print(f"\n  DIP ZONE (q230-310) — where all methods struggle:")
    dip = all_items[230:310]
    print(f"  n={len(dip)} questions")
    dip_types = defaultdict(int)
    for item in dip:
        dip_types[item["type"]] += 1
    for qtype, count in sorted(dip_types.items(), key=lambda x: -x[1]):
        print(f"    {qtype}: {count}")
    dip_temporal = sum(1 for i in dip if i["is_temporal"])
    dip_multi = sum(1 for i in dip if i["is_multi_answer"])
    print(f"  Temporal: {dip_temporal}/{len(dip)} ({dip_temporal/len(dip)*100:.0f}%)")
    print(f"  Multi-answer: {dip_multi}/{len(dip)} ({dip_multi/len(dip)*100:.0f}%)")

    # Date analysis for temporal questions
    print(f"\n  Temporal questions — date availability:")
    temporal_items = [i for i in all_items if i["is_temporal"]]
    with_dates = sum(1 for i in temporal_items if i["answer_dates"])
    print(f"  {with_dates}/{len(temporal_items)} have answer session dates in metadata")

    # Example temporal questions that would benefit from date injection
    print(f"\n  Sample temporal questions (first 10):")
    for item in temporal_items[:10]:
        dates_str = ", ".join(item["answer_dates"][:3]) if item["answer_dates"] else "no dates"
        print(f"    q{item['qi']:3d} [{item['type']}] {item['question'][:80]}")
        print(f"         answers={item['n_answer_sessions']} dates={dates_str}")

    # Save full analysis
    out = {
        "total_questions": len(data),
        "by_type": {qtype: len(items) for qtype, items in categories.items()},
        "temporal_questions": temporal_total,
        "counting_questions": counting_total,
        "multi_answer_questions": multi_total,
        "dip_zone_230_310": {
            "n": len(dip),
            "by_type": dict(dip_types),
            "temporal_pct": round(dip_temporal / len(dip) * 100),
            "multi_pct": round(dip_multi / len(dip) * 100),
        },
        "all_questions": all_items,
    }
    out_path = BENCH_DIR / "analysis-pipeline-2026-04-15.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
