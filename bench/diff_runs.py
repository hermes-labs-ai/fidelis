"""
Compare two per-question run logs.

Usage:
  python bench/diff_runs.py bench/runs/baseline/per_question.json bench/runs/experiment/per_question.json
  python bench/diff_runs.py <run_id_a> <run_id_b>   # shorthand: looks in bench/runs/<id>/per_question.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

BENCH_DIR = Path(__file__).parent


def load_run(path_or_id: str) -> list[dict]:
    p = Path(path_or_id)
    if not p.exists():
        p = BENCH_DIR / "runs" / path_or_id / "per_question.json"
    if not p.exists():
        print(f"Not found: {path_or_id}")
        sys.exit(1)
    return json.load(open(p))


def main():
    if len(sys.argv) != 3:
        print("Usage: python bench/diff_runs.py <run_a> <run_b>")
        sys.exit(1)

    run_a = load_run(sys.argv[1])
    run_b = load_run(sys.argv[2])

    if len(run_a) != len(run_b):
        print(f"Warning: different question counts ({len(run_a)} vs {len(run_b)})")

    # Index by qid
    a_by_qid = {q["qid"]: q for q in run_a}
    b_by_qid = {q["qid"]: q for q in run_b}
    common = set(a_by_qid) & set(b_by_qid)

    # Compare S2 R@1 (final output)
    wins = []       # A miss, B hit
    losses = []     # A hit, B miss
    both_right = []
    both_wrong = []

    for qid in sorted(common, key=lambda x: a_by_qid[x]["qi"]):
        a = a_by_qid[qid]
        b = b_by_qid[qid]
        a_hit = a["s2_hit_at_1"]
        b_hit = b["s2_hit_at_1"]

        if not a_hit and b_hit:
            wins.append((a, b))
        elif a_hit and not b_hit:
            losses.append((a, b))
        elif a_hit and b_hit:
            both_right.append((a, b))
        else:
            both_wrong.append((a, b))

    n = len(common)
    a_r1 = (len(wins) + len(losses) + len(both_right)) and (len(both_right) + len(losses)) / n
    b_r1 = (len(both_right) + len(wins)) / n

    print(f"{'='*70}")
    print(f"  Run A: {sys.argv[1]}")
    print(f"  Run B: {sys.argv[2]}")
    print(f"  Common questions: {n}")
    print(f"{'='*70}\n")

    print(f"  R@1: A={len(both_right)+len(losses)}/{n} ({(len(both_right)+len(losses))/n:.1%})  "
          f"B={len(both_right)+len(wins)}/{n} ({(len(both_right)+len(wins))/n:.1%})")
    print(f"  Wins (A miss → B hit):   {len(wins)}")
    print(f"  Losses (A hit → B miss): {len(losses)}")
    print(f"  Both right:              {len(both_right)}")
    print(f"  Both wrong:              {len(both_wrong)}")
    print(f"  Net delta:               {len(wins) - len(losses):+d} questions")

    # Break down by qtype
    def breakdown(items, label):
        if not items:
            return
        print(f"\n  {label} ({len(items)}):")
        by_type = defaultdict(list)
        for a, b in items:
            by_type[a["qtype"]].append(a)
        for qtype, qs in sorted(by_type.items(), key=lambda x: -len(x[1])):
            multi = sum(1 for q in qs if q["is_multi_answer"])
            print(f"    {qtype:35s} n={len(qs):3d}  multi={multi:3d}")

    breakdown(wins, "WINS (A miss → B hit)")
    breakdown(losses, "LOSSES (A hit → B miss)")
    breakdown(both_wrong, "BOTH WRONG")

    # S1 vs S2 analysis within each run
    print(f"\n{'─'*70}")
    print(f"  S1 vs S2 within Run A:")
    a_s1_wins = sum(1 for q in run_a if not q["s1_hit_at_1"] and q["s2_hit_at_1"])
    a_s1_losses = sum(1 for q in run_a if q["s1_hit_at_1"] and not q["s2_hit_at_1"])
    print(f"    Filter wins: {a_s1_wins}  Filter losses: {a_s1_losses}  Net: {a_s1_wins - a_s1_losses:+d}")

    # Break filter wins/losses by qtype
    print(f"\n    Filter WINS by qtype (S1 miss → S2 hit):")
    fw_by_type = defaultdict(int)
    for q in run_a:
        if not q["s1_hit_at_1"] and q["s2_hit_at_1"]:
            fw_by_type[q["qtype"]] += 1
    for qtype, count in sorted(fw_by_type.items(), key=lambda x: -x[1]):
        print(f"      {qtype:35s} {count:3d}")

    print(f"\n    Filter LOSSES by qtype (S1 hit → S2 miss):")
    fl_by_type = defaultdict(int)
    for q in run_a:
        if q["s1_hit_at_1"] and not q["s2_hit_at_1"]:
            fl_by_type[q["qtype"]] += 1
    for qtype, count in sorted(fl_by_type.items(), key=lambda x: -x[1]):
        print(f"      {qtype:35s} {count:3d}")

    # Multi-answer analysis
    print(f"\n{'─'*70}")
    print(f"  Multi-answer analysis (Run A):")
    multi_qs = [q for q in run_a if q["is_multi_answer"]]
    single_qs = [q for q in run_a if not q["is_multi_answer"]]
    multi_s1_r1 = sum(q["s1_hit_at_1"] for q in multi_qs) / len(multi_qs) if multi_qs else 0
    multi_s2_r1 = sum(q["s2_hit_at_1"] for q in multi_qs) / len(multi_qs) if multi_qs else 0
    single_s1_r1 = sum(q["s1_hit_at_1"] for q in single_qs) / len(single_qs) if single_qs else 0
    single_s2_r1 = sum(q["s2_hit_at_1"] for q in single_qs) / len(single_qs) if single_qs else 0
    print(f"    Multi-answer ({len(multi_qs)}q):  S1 R@1={multi_s1_r1:.1%}  S2 R@1={multi_s2_r1:.1%}  delta={multi_s2_r1-multi_s1_r1:+.1%}")
    print(f"    Single-answer ({len(single_qs)}q): S1 R@1={single_s1_r1:.1%}  S2 R@1={single_s2_r1:.1%}  delta={single_s2_r1-single_s1_r1:+.1%}")

    # Print the first 10 both_wrong for diagnosis
    if both_wrong:
        print(f"\n{'─'*70}")
        print(f"  Sample BOTH WRONG (first 10):")
        for a, b in both_wrong[:10]:
            print(f"    q{a['qi']:3d} [{a['qtype']}] {a['question'][:70]}")
            print(f"         gold={a['gold_session_ids'][:3]}  s1_top5={a['s1_top5_ids'][:3]}")


if __name__ == "__main__":
    main()
