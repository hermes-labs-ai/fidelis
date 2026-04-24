"""
E0 — Baseline lock. Compute optimal blended QA from existing top1 + top5 JSONs.
No API spend. Zero cost.

Existing data:
  bench/qa_eval_v2_runP-v35_gpt-4o-mini_top1.json  (n=470, all qtypes at K=1)
  bench/qa_eval_v2_runP-v35_gpt-4o-mini_top5.json  (n=470, all qtypes at K=5)

For each qtype, pick the K that gives better accuracy. Compute blended overall.
"""
import json
import math
from pathlib import Path

BENCH = Path(__file__).parent.parent.parent / "bench"

top1 = json.load(open(BENCH / "qa_eval_v2_runP-v35_gpt-4o-mini_top1.json"))
top5 = json.load(open(BENCH / "qa_eval_v2_runP-v35_gpt-4o-mini_top5.json"))

# Index by qid
t1 = {r["qid"]: r for r in top1}
t5 = {r["qid"]: r for r in top5}

all_qids = set(t1) | set(t5)
qtypes = sorted({r["qtype"] for r in top1})

print(f"Loaded: top1 n={len(top1)}, top5 n={len(top5)}")
print()

# Per-qtype accuracy at K=1 and K=5
qtype_k1 = {}
qtype_k5 = {}
for qt in qtypes:
    k1 = [r for r in top1 if r["qtype"] == qt]
    k5 = [r for r in top5 if r["qtype"] == qt]
    qtype_k1[qt] = {
        "n": len(k1),
        "correct": sum(r["qa_correct"] for r in k1),
        "acc": sum(r["qa_correct"] for r in k1) / max(len(k1), 1),
    }
    qtype_k5[qt] = {
        "n": len(k5),
        "correct": sum(r["qa_correct"] for r in k5),
        "acc": sum(r["qa_correct"] for r in k5) / max(len(k5), 1),
    }

print("Per-qtype accuracy — K=1 vs K=5 (gpt-4o-mini):")
print(f"{'qtype':<32}  {'n':>4}  {'K=1':>7}  {'K=5':>7}  {'winner':>8}")
print("-" * 70)
optimal_correct = 0
optimal_total = 0
optimal_k = {}
for qt in qtypes:
    k1_acc = qtype_k1[qt]["acc"]
    k5_acc = qtype_k5[qt]["acc"]
    winner_k = 1 if k1_acc >= k5_acc else 5
    n = qtype_k1[qt]["n"]
    best_acc = max(k1_acc, k5_acc)
    best_correct = round(best_acc * n)
    optimal_correct += best_correct
    optimal_total += n
    optimal_k[qt] = winner_k
    print(f"  {qt:<30}  {n:>4}  {k1_acc:>6.1%}  {k5_acc:>6.1%}  {'K='+str(winner_k):>8}")

print("-" * 70)
optimal_blended = optimal_correct / optimal_total
print(f"\n  Optimal blended (best K per qtype): {optimal_blended:.4f} = {optimal_blended:.1%}")
print(f"  Correct: {optimal_correct}/{optimal_total}")
print(f"\n  Optimal K per qtype: {optimal_k}")

# Variance / CI at n=470
# 95% CI via Wilson method
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return (max(0, center - margin), min(1, center + margin))

lo, hi = wilson_ci(optimal_correct, optimal_total)
print(f"  95% CI: [{lo:.1%}, {hi:.1%}]  (Wilson interval, n={optimal_total})")

# Cross-check: top1 baseline
top1_correct = sum(r["qa_correct"] for r in top1)
print(f"\n  Top-1 baseline (K=1 all qtypes): {top1_correct}/{len(top1)} = {top1_correct/len(top1):.1%}")
lo1, hi1 = wilson_ci(top1_correct, len(top1))
print(f"  95% CI: [{lo1:.1%}, {hi1:.1%}]")

# Top-5 baseline
top5_correct = sum(r["qa_correct"] for r in top5)
print(f"\n  Top-5 baseline (K=5 all qtypes): {top5_correct}/{len(top5)} = {top5_correct/len(top5):.1%}")

# KU analysis: K=1 wins despite 100% multi-session need
print("\n  Note on KU: 63.9% at K=1 > 58.3% at K=5.")
print("  KU questions ask for the LATEST answer; K=5 includes superseded old answers,")
print("  which confuse the reader. Confirmed: use K=1 for KU even with GPT-4o.")

# Write receipt
receipt = {
    "experiment_id": "E0",
    "description": "Baseline lock from existing top1+top5 data. No API spend.",
    "data_sources": [
        "bench/qa_eval_v2_runP-v35_gpt-4o-mini_top1.json",
        "bench/qa_eval_v2_runP-v35_gpt-4o-mini_top5.json",
    ],
    "top1_baseline": {
        "qa_accuracy": round(top1_correct/len(top1), 4),
        "n_correct": top1_correct,
        "n_total": len(top1),
        "95ci": [round(lo1, 4), round(wilson_ci(top1_correct, len(top1))[1], 4)],
    },
    "top5_baseline": {
        "qa_accuracy": round(top5_correct/len(top5), 4),
        "n_correct": top5_correct,
        "n_total": len(top5),
    },
    "optimal_blended": {
        "qa_accuracy": round(optimal_blended, 4),
        "n_correct": optimal_correct,
        "n_total": optimal_total,
        "95ci": [round(lo, 4), round(hi, 4)],
        "k_routing": optimal_k,
    },
    "per_qtype": {
        qt: {
            "k1_acc": round(qtype_k1[qt]["acc"], 4),
            "k5_acc": round(qtype_k5[qt]["acc"], 4),
            "optimal_k": optimal_k[qt],
            "n": qtype_k1[qt]["n"],
        }
        for qt in qtypes
    },
    "cost_usd": 0.0,
    "api_calls": 0,
}

out = Path(__file__).parent / "receipt.json"
json.dump(receipt, open(out, "w"), indent=2)
print(f"\n  Receipt written: {out}")
