"""
Ablation harness for v33 component attribution.

Runs 4 ablation benchmarks sequentially, each with one component disabled:
  K: --no-temporal-boost   (measure s8-socratic lift in isolation)
  L: --no-socratic         (measure temporal-boost lift in isolation)
  M: --no-verify-guard     (measure verify-guard contribution)
  N: --no-flagship         (measure flagship contribution + cost baseline)

Each run outputs to bench/runs/<run-id>/per_question.json + aggregate.json.
Final summary printed + written to bench/ABLATION_RESULTS.md.

Usage:
  DASHSCOPE_API_KEY=sk-xxx python3 bench/run_ablations.py [--resume] [--only K,L,M,N]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).parent
ABLATE_SCRIPT = BENCH_DIR / "longmemeval_combined_pipeline_v33_ablate.py"

# Default data dir: check sibling LongMemEval project
_DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "LongMemEval" / "data"
if not _DEFAULT_DATA_DIR.exists():
    # LongMemEval dataset must be cloned alongside this repo; pass --data_dir if
    # your layout differs: e.g. python3 bench/run_ablations.py --data_dir /path/to/LongMemEval/data
    _DEFAULT_DATA_DIR = Path.home() / "LongMemEval" / "data"

# Full v33 reference
FULL_V33 = {
    "run_id": "runJ-v33",
    "r1": 0.9511,
    "label": "Full system (runJ-v33)",
}

# Ablation definitions
ABLATIONS = [
    {
        "run_id": "runK-ablate-tempboost",
        "flag": "--no-temporal-boost",
        "label": "-temporal-boost (runK)",
        "interpretation": "temporal-boost contributes X pp; s8-socratic isolated",
    },
    {
        "run_id": "runL-ablate-socratic",
        "flag": "--no-socratic",
        "label": "-s8-socratic / baseline filter (runL)",
        "interpretation": "s8-socratic contributes X pp; temporal-boost isolated",
    },
    {
        "run_id": "runM-ablate-guard",
        "flag": "--no-verify-guard",
        "label": "-verify-guard (runM)",
        "interpretation": "verify-guard marginal contribution",
    },
    {
        "run_id": "runN-ablate-flagship",
        "flag": "--no-flagship",
        "label": "-flagship / qwen-max off (runN)",
        "interpretation": "flagship marginal contribution + cost sensitivity",
    },
]


def run_ablation(ablation: dict, resume: bool, data_dir: str | None) -> dict | None:
    """Run a single ablation. Returns aggregate dict or None on crash."""
    run_id = ablation["run_id"]
    flag = ablation["flag"]

    run_dir = BENCH_DIR / "runs" / run_id
    agg_path = run_dir / "aggregate.json"

    # Check if already complete
    if agg_path.exists():
        existing = json.load(open(agg_path))
        n_q = existing.get("questions_evaluated", 0)
        if n_q >= 470:
            print(f"\n[harness] {run_id} already complete ({n_q} questions). Loading.")
            return existing
        elif resume:
            print(f"\n[harness] {run_id} partial ({n_q}/470). Resuming.")
        else:
            print(f"\n[harness] {run_id} partial ({n_q}/470). Starting fresh (use --resume to continue).")

    print(f"\n{'='*70}")
    print(f"[harness] Starting ablation: {run_id}")
    print(f"[harness] Flag: {flag}")
    print(f"[harness] Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    cmd = [
        sys.executable,
        str(ABLATE_SCRIPT),
        "--split", "s",
        "--run-id", run_id,
        flag,
    ]
    if resume:
        cmd.append("--resume")
    if data_dir:
        cmd.extend(["--data_dir", data_dir])

    env = os.environ.copy()
    if not env.get("DASHSCOPE_API_KEY"):
        # Try hardcoded key from task spec as fallback
        env["DASHSCOPE_API_KEY"] = os.environ.get("DASHSCOPE_API_KEY", "")

    t_start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            timeout=1800,  # 30 min max per ablation
        )
        elapsed = time.time() - t_start
        if proc.returncode != 0:
            print(f"\n[harness] ERROR: {run_id} exited with code {proc.returncode}")
            print(f"[harness] Elapsed: {elapsed:.0f}s")
            return None
    except subprocess.TimeoutExpired:
        print(f"\n[harness] TIMEOUT: {run_id} exceeded 30 min limit")
        return None
    except Exception as e:
        print(f"\n[harness] CRASH: {run_id} — {e}")
        return None

    elapsed = time.time() - t_start
    print(f"\n[harness] {run_id} completed in {elapsed:.0f}s")

    if agg_path.exists():
        return json.load(open(agg_path))
    else:
        print(f"[harness] WARNING: aggregate.json not found for {run_id}")
        return None


def build_qtype_table(results: list[dict], labels: list[str]) -> str:
    """Build per-qtype markdown table."""
    # Collect all qtypes
    all_qtypes = set()
    for r in results:
        if r:
            all_qtypes.update(r.get("qtype_breakdown", {}).keys())
    all_qtypes = sorted(all_qtypes)

    lines = []
    # Header
    header = f"| {'qtype':<30} | {'Full v33':>8} |"
    for label in labels:
        short = label[:18]
        header += f" {short:>18} |"
    lines.append(header)
    lines.append("|" + "-" * 32 + "|" + ("-" * 10 + "|") * (1 + len(labels)))

    for qt in all_qtypes:
        row = f"| {qt:<30} |"
        # Full v33 from runJ
        runj_path = BENCH_DIR / "runs" / "runJ-v33" / "aggregate.json"
        if runj_path.exists():
            runj_agg = json.load(open(runj_path))
            qt_data = runj_agg.get("qtype_breakdown", {}).get(qt, {})
            r1 = qt_data.get("r1", None)
            row += f" {r1:.1%} (n={qt_data.get('n', 0)})" if r1 is not None else " N/A      |"
        else:
            row += " N/A      "
        row += " |"
        # Each ablation
        for r in results:
            if r is None:
                row += " CRASHED           |"
                continue
            qt_data = r.get("qtype_breakdown", {}).get(qt, {})
            r1 = qt_data.get("r1", None)
            delta = qt_data.get("delta_vs_v33", "N/A")
            if r1 is not None:
                row += f" {r1:.1%} ({delta:>6})    |"
            else:
                row += " N/A               |"
        lines.append(row)

    return "\n".join(lines)


def estimate_costs(results: list[dict], labels: list[str]) -> str:
    """Estimate per-ablation costs based on filter call counts and timing."""
    # Rough cost model:
    # qwen-turbo: ~$0.001/query for filter call (~1K tokens in, ~200 out)
    # qwen-max: ~$0.06/query for flagship call (~3K tokens in, ~100 out)
    # verify-guard: ~$0.001/query (extra turbo call)
    TURBO_COST_PER_CALL = 0.001  # rough $/call
    MAX_COST_PER_CALL = 0.06     # rough $/call

    lines = []
    lines.append("| Component | LLM calls | Est. cost | Cost/query |")
    lines.append("|-----------|-----------|-----------|------------|")

    for i, r in enumerate(results):
        if r is None:
            lines.append(f"| {labels[i]:<20} | CRASHED | - | - |")
            continue
        fm = r.get("filter_methods", {})
        llm_calls = fm.get("llm", 0) + fm.get("llm_guarded_s1", 0)
        flagship_calls = fm.get("flagship", 0) + fm.get("flagship_llm", 0)
        verify_calls = fm.get("llm_guarded_s1", 0)  # verify fires on guarded questions
        n_q = r.get("questions_evaluated", 470)

        turbo_cost = llm_calls * TURBO_COST_PER_CALL
        max_cost = flagship_calls * MAX_COST_PER_CALL
        total_est = turbo_cost + max_cost
        per_q = total_est / n_q if n_q else 0

        lines.append(
            f"| {labels[i][:20]:<20} | turbo={llm_calls} flag={flagship_calls} | ${total_est:.2f} | ${per_q:.4f} |"
        )

    return "\n".join(lines)


def write_results_md(ablation_results: list[dict | None], ablations: list[dict]) -> None:
    """Write ABLATION_RESULTS.md with full attribution table."""
    out_path = BENCH_DIR / "ABLATION_RESULTS.md"
    full_r1 = FULL_V33["r1"]

    lines = [
        "# Ablation Results — v33 Component Attribution",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d')}",
        f"**Full system (runJ-v33)**: {full_r1:.2%} R@1 on LongMemEval_S (470 questions)",
        f"**Baseline (runC-guard)**: 94.00% R@1",
        f"**Delta to attribute**: +1.11pp",
        "",
        "## Attribution Table",
        "",
        "| Variant | R@1 | Delta from full | Interpretation |",
        "|---------|-----|-----------------|----------------|",
        f"| Full system (runJ-v33) | {full_r1:.2%} | 0 pp | baseline claim |",
    ]

    labels = []
    for i, (ablation, result) in enumerate(zip(ablations, ablation_results)):
        label = ablation["label"]
        labels.append(label)
        if result is None:
            lines.append(f"| {label} | CRASHED | - | run failed |")
            continue
        r1 = result.get("stage2_r1", 0)
        delta = r1 - full_r1
        n_q = result.get("questions_evaluated", 0)
        interp = ablation["interpretation"].replace("X pp", f"{abs(delta):.2f} pp")
        lines.append(f"| {label} | {r1:.2%} | {delta:+.2f} pp | {interp} |")

    lines += [
        "",
        "## Per-qtype Breakdown",
        "",
        "Delta vs full v33 (95.11%) per question type.",
        "",
    ]
    lines.append(build_qtype_table(ablation_results, labels))

    lines += [
        "",
        "## Cost per Component",
        "",
        "Rough estimates based on filter call counts.",
        "",
    ]
    lines.append(estimate_costs(ablation_results, labels))

    lines += [
        "",
        "## Lift per Dollar Analysis",
        "",
    ]

    # Compute lift-per-dollar ranking
    lpd_rows = []
    for i, (ablation, result) in enumerate(zip(ablations, ablation_results)):
        if result is None:
            continue
        r1 = result.get("stage2_r1", 0)
        delta = full_r1 - r1  # positive = component helped (removing it hurt)
        fm = result.get("filter_methods", {})
        llm_calls = fm.get("llm", 0) + fm.get("llm_guarded_s1", 0)
        flagship_calls = fm.get("flagship", 0) + fm.get("flagship_llm", 0)
        cost = llm_calls * 0.001 + flagship_calls * 0.06
        if cost > 0 and delta != 0:
            lpd = abs(delta) / cost
            lpd_rows.append((ablation["label"], delta, cost, lpd))
        else:
            lpd_rows.append((ablation["label"], delta, cost, 0))

    lpd_rows.sort(key=lambda x: x[3], reverse=True)
    lines.append("| Component | Lift (pp) | Est. run cost | Lift/$ |")
    lines.append("|-----------|-----------|---------------|--------|")
    for label, delta, cost, lpd in lpd_rows:
        direction = "helped" if delta > 0 else "HURT (net negative)" if delta < 0 else "neutral"
        lines.append(f"| {label[:30]} | {delta*100:+.2f} | ${cost:.2f} | {lpd:.4f} | {direction} |")

    lines += [
        "",
        "## Which Component Fails Gracefully?",
        "",
    ]
    for i, (ablation, result) in enumerate(zip(ablations, ablation_results)):
        if result is None:
            continue
        r1 = result.get("stage2_r1", 0)
        delta = r1 - full_r1
        flag = ablation["flag"]
        if delta >= -0.005:
            verdict = "SAFE to drop — <0.5pp loss"
        elif delta >= -0.01:
            verdict = "marginal — 0.5-1pp loss, consider tradeoff"
        else:
            verdict = "ESSENTIAL — >1pp loss without it"
        lines.append(f"- `{flag}`: {delta:+.2f} pp → **{verdict}**")

    lines += [
        "",
        "## Surprising Findings",
        "",
    ]
    surprises = []
    for i, (ablation, result) in enumerate(zip(ablations, ablation_results)):
        if result is None:
            continue
        r1 = result.get("stage2_r1", 0)
        delta = r1 - full_r1
        if delta > 0:
            surprises.append(
                f"- **{ablation['flag']}**: Removing this component IMPROVED R@1 by {delta:+.2%}. "
                f"Component may be net-harmful in this configuration."
            )
        # Check per-qtype surprises
        for qt, qt_data in result.get("qtype_breakdown", {}).items():
            qt_delta = qt_data.get("delta_vs_v33", "N/A")
            if isinstance(qt_delta, str) and qt_delta.startswith("+") and float(qt_delta.replace("%", "")) > 2:
                surprises.append(
                    f"- **{ablation['flag']}** on `{qt}`: removing helped by {qt_delta} — "
                    f"component hurts this qtype."
                )

    if surprises:
        lines.extend(surprises)
    else:
        lines.append("- No surprise inversions detected. All ablations degraded or held.")

    lines += [
        "",
        "## Run Details",
        "",
        "| Run ID | Questions | Wall time | Flag |",
        "|--------|-----------|-----------|------|",
    ]
    for ablation, result in zip(ablations, ablation_results):
        if result is None:
            lines.append(f"| {ablation['run_id']} | CRASHED | - | {ablation['flag']} |")
        else:
            n_q = result.get("questions_evaluated", 0)
            t_s = result.get("total_time_s", 0)
            lines.append(f"| {ablation['run_id']} | {n_q} | {t_s:.0f}s | {ablation['flag']} |")

    lines.append("")
    content = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(content)
    print(f"\n[harness] ABLATION_RESULTS.md written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume partial runs")
    parser.add_argument("--only", default="", help="Comma-separated run IDs to run (e.g. K,L or runK-ablate-tempboost,runL-ablate-socratic)")
    parser.add_argument("--data_dir", default=str(_DEFAULT_DATA_DIR), help="Data directory (default: auto-detected)")
    args = parser.parse_args()

    # Filter ablations if --only specified
    ablations_to_run = ABLATIONS
    if args.only:
        tokens = [t.strip() for t in args.only.split(",")]
        ablations_to_run = []
        for a in ABLATIONS:
            for tok in tokens:
                if tok in a["run_id"] or tok == a["run_id"]:
                    ablations_to_run.append(a)
                    break

    if not ABLATE_SCRIPT.exists():
        print(f"ERROR: ablation script not found: {ABLATE_SCRIPT}")
        sys.exit(1)

    print(f"[harness] v33 Ablation Suite")
    print(f"[harness] Running {len(ablations_to_run)} ablations sequentially")
    print(f"[harness] Resume: {args.resume}")
    print(f"[harness] Script: {ABLATE_SCRIPT}")
    print()

    results: list[dict | None] = []
    for ablation in ablations_to_run:
        result = run_ablation(ablation, args.resume, args.data_dir)
        results.append(result)
        if result:
            r1 = result.get("stage2_r1", 0)
            delta = r1 - FULL_V33["r1"]
            print(f"\n[harness] {ablation['run_id']}: R@1={r1:.2%}  delta={delta:+.2f}pp vs full v33")
        else:
            print(f"\n[harness] {ablation['run_id']}: CRASHED or INCOMPLETE")

    # Build results for ALL ablations (even ones not run this session)
    all_results = []
    for ablation in ABLATIONS:
        if ablation in ablations_to_run:
            idx = ablations_to_run.index(ablation)
            all_results.append(results[idx])
        else:
            # Try loading existing
            agg_path = BENCH_DIR / "runs" / ablation["run_id"] / "aggregate.json"
            if agg_path.exists():
                all_results.append(json.load(open(agg_path)))
            else:
                all_results.append(None)

    write_results_md(all_results, ABLATIONS)

    # Print headline summary
    print(f"\n{'='*70}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Full system (runJ-v33): {FULL_V33['r1']:.2%}")
    print()
    for ablation, result in zip(ABLATIONS, all_results):
        if result is None:
            print(f"  {ablation['run_id']:<30}: CRASHED")
        else:
            r1 = result.get("stage2_r1", 0)
            delta = r1 - FULL_V33["r1"]
            print(f"  {ablation['run_id']:<30}: {r1:.2%}  ({delta:+.2f}pp)")

    largest = None
    largest_delta = 0
    for ablation, result in zip(ABLATIONS, all_results):
        if result:
            delta = FULL_V33["r1"] - result.get("stage2_r1", 0)
            if delta > largest_delta:
                largest_delta = delta
                largest = ablation["run_id"]

    if largest:
        print(f"\n  Largest contributor: {largest} ({largest_delta:+.2f}pp lift)")
    print(f"\n  Results: {BENCH_DIR}/ABLATION_RESULTS.md")


if __name__ == "__main__":
    main()
