"""
Step 3: Build hardset.json — questions missed by BOTH S1 and routed S2.
These are the fixed regression set for future phases.
"""
import json
from pathlib import Path

BENCH = Path(__file__).parent.parent

pq = json.load(open(BENCH / "runs/step1-gapfix/per_question.json"))
data = json.load(open(Path.home() / "Documents/projects/LongMemEval/data/longmemeval_s_cleaned.json"))
data = [e for e in data if "_abs" not in e["question_id"]]

# Both wrong = S1 miss AND S2 (routed) miss
hardset = []
for q in pq:
    if not q["s1_hit_at_1"] and not q["s2_hit_at_1"]:
        entry = data[q["qi"]]
        # Classify failure mode
        question = entry["question"].lower()
        n_gold = len(entry["answer_session_ids"])

        if n_gold > 1 and any(p in question for p in ["order", "first", "before", "after", "earliest"]):
            mode = "temporal_multi_hop"
        elif n_gold > 1:
            mode = "multi_session_aggregation"
        elif any(p in question for p in ["you told", "you said", "you suggested", "you mentioned"]):
            mode = "assistant_reference"
        elif any(p in question for p in ["preference", "favorite", "like", "enjoy"]):
            mode = "preference_recall"
        elif any(p in question for p in ["update", "change", "now", "current"]):
            mode = "knowledge_update"
        else:
            mode = "other"

        hardset.append({
            "qid": q["qid"],
            "qi": q["qi"],
            "qtype": q["qtype"],
            "question": entry["question"],
            "gold_session_ids": entry["answer_session_ids"],
            "n_gold": n_gold,
            "s1_top5_ids": q["s1_top5_ids"],
            "s2_top5_ids": q["s2_top5_ids"],
            "route_decision": q["route_decision"],
            "failure_mode_note": mode,
            "s1_hit_at_5": q["s1_hit_at_5"],
            "s2_hit_at_5": q["s2_hit_at_5"],
        })

# Save
out_path = BENCH / "hardset.json"
with open(out_path, "w") as f:
    json.dump(hardset, f, indent=2)

# Report
from collections import Counter
modes = Counter(h["failure_mode_note"] for h in hardset)
types = Counter(h["qtype"] for h in hardset)
in_s1_top5 = sum(1 for h in hardset if h["s1_hit_at_5"])

print(f"HARDSET: {len(hardset)} questions missed by both S1 and routed S2")
print(f"  Gold in S1 top-5: {in_s1_top5}/{len(hardset)} ({in_s1_top5/len(hardset)*100:.0f}%)")
print()
print("By failure mode:")
for mode, count in modes.most_common():
    print(f"  {mode:30s} {count:3d}")
print()
print("By question type:")
for qtype, count in types.most_common():
    print(f"  {qtype:30s} {count:3d}")
print()
print("Sample hard questions:")
for h in hardset[:10]:
    print(f"  q{h['qi']:3d} [{h['failure_mode_note']}] {h['question'][:80]}")

print(f"\nSaved to {out_path}")
