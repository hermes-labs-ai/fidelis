"""
cogito benchmark — compares two-stage /recall against naive /query.

Measures Recall@1, Recall@3, MRR, precision, and latency for each method
on a fixed query set. Requires a running cogito server and a test cases file.

Usage:
    python bench/benchmark.py                          # uses bench/cases.json
    python bench/benchmark.py --cases my_cases.json
    python bench/benchmark.py --port 19420 --verbose
    python bench/benchmark.py --methods recall,query   # only run these

Test case format (bench/cases.json):
    [
      {
        "query": "lintlang release version",
        "expected": ["lintlang", "v0.", "release", "version"],   # ANY match = hit
        "notes": "optional description"
      },
      ...
    ]

A result is a HIT if ANY expected keyword appears (case-insensitive) in the
returned memory text. This is intentionally lenient — the goal is to measure
whether relevant memories surface, not exact string matching.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── types ──────────────────────────────────────────────────────────────────

@dataclass
class Case:
    query: str
    expected: list[str]
    notes: str = ""
    difficulty: str = ""


@dataclass
class Result:
    query: str
    method: str
    memories: list[dict]
    api_method: str       # method tag returned by server
    latency_ms: float
    hit_at_1: bool = False
    hit_at_3: bool = False
    rr: float = 0.0       # reciprocal rank (for MRR)
    matched_keyword: str = ""

    def rank_of_first_hit(self) -> Optional[int]:
        for i, m in enumerate(self.memories, 1):
            if _is_hit(m["text"], self.matched_keyword):
                return i
        return None


# ── helpers ────────────────────────────────────────────────────────────────

def _is_hit(text: str, keywords: str | list[str]) -> bool:
    if isinstance(keywords, str):
        keywords = [keywords]
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def _first_hit_keyword(text: str, keywords: list[str]) -> str:
    t = text.lower()
    for kw in keywords:
        if kw.lower() in t:
            return kw
    return ""


def _post(base_url: str, path: str, payload: dict, timeout: int = 30) -> tuple[dict, float]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        latency = (time.monotonic() - t0) * 1000
        return result, latency
    except urllib.error.URLError as e:
        print(f"\nError: server not reachable at {base_url} — {e}", file=sys.stderr)
        sys.exit(1)


# ── runner ─────────────────────────────────────────────────────────────────

def run_case(base_url: str, case: Case, method: str, limit: int) -> Result:
    if method == "recall":
        resp, latency = _post(base_url, "/recall", {"text": case.query, "limit": limit})
    elif method == "recall_b":
        resp, latency = _post(base_url, "/recall_b", {"text": case.query, "limit": limit})
    else:
        resp, latency = _post(base_url, "/query", {"text": case.query, "limit": limit})

    memories = resp.get("memories", [])
    api_method = resp.get("method", method)

    result = Result(
        query=case.query,
        method=method,
        memories=memories,
        api_method=api_method,
        latency_ms=latency,
    )

    # Score
    for i, m in enumerate(memories[:3], 1):
        kw = _first_hit_keyword(m["text"], case.expected)
        if kw:
            result.matched_keyword = kw
            if i == 1:
                result.hit_at_1 = True
            result.hit_at_3 = True
            result.rr = 1.0 / i
            break

    return result


# ── reporting ──────────────────────────────────────────────────────────────

def print_results(results_by_method: dict[str, list[Result]], cases: list[Case], verbose: bool):
    n = len(cases)
    methods = list(results_by_method.keys())
    hard_indices = [i for i, c in enumerate(cases) if c.difficulty == "hard"]
    n_hard = len(hard_indices)

    print(f"\n{'─'*70}")
    print(f"  cogito benchmark  —  {n} queries  ({n_hard} hard semantic)")
    print(f"{'─'*70}\n")

    # Per-query detail
    if verbose:
        for i, case in enumerate(cases):
            print(f"  [{i+1}] {case.query}")
            if case.notes:
                print(f"       note: {case.notes}")
            for method in methods:
                r = results_by_method[method][i]
                hit1 = "✅" if r.hit_at_1 else ("🟡" if r.hit_at_3 else "❌")
                first = r.memories[0]["text"][:60] if r.memories else "(no results)"
                kw_tag = f" ← matched '{r.matched_keyword}'" if r.matched_keyword else ""
                print(f"       {method:8s} {hit1}  {r.latency_ms:5.0f}ms  {first}{kw_tag}")
            print()

    # Aggregate metrics
    print(f"  {'Metric':<22}", end="")
    for m in methods:
        print(f"  {m:>10}", end="")
    print()
    print(f"  {'─'*22}", end="")
    for _ in methods:
        print(f"  {'─'*10}", end="")
    print()

    metrics = {
        "Recall@1":      lambda rs: sum(r.hit_at_1 for r in rs) / n,
        "Recall@3":      lambda rs: sum(r.hit_at_3 for r in rs) / n,
        "MRR":           lambda rs: sum(r.rr for r in rs) / n,
        "Avg latency ms": lambda rs: sum(r.latency_ms for r in rs) / n,
        "Filter used %": lambda rs: sum(1 for r in rs if r.api_method == "filter") / n,
    }

    for label, fn in metrics.items():
        print(f"  {label:<22}", end="")
        for method in methods:
            val = fn(results_by_method[method])
            if "ms" in label:
                print(f"  {val:>10.0f}", end="")
            elif "%" in label:
                print(f"  {val:>9.0%} ", end="")
            else:
                print(f"  {val:>10.3f}", end="")
        print()

    # Hard-case breakdown
    if n_hard > 0:
        print(f"\n  Hard cases only ({n_hard} semantic-gap queries):")
        hard_metrics = {
            "Hard Recall@1":  lambda rs: sum(rs[i].hit_at_1 for i in hard_indices) / n_hard,
            "Hard Recall@3":  lambda rs: sum(rs[i].hit_at_3 for i in hard_indices) / n_hard,
            "Hard MRR":       lambda rs: sum(rs[i].rr for i in hard_indices) / n_hard,
        }
        for label, fn in hard_metrics.items():
            print(f"  {label:<22}", end="")
            for method in methods:
                val = fn(results_by_method[method])
                print(f"  {val:>10.3f}", end="")
            print()

    print(f"\n{'─'*70}\n")

    # Delta summary
    sign = lambda x: "+" if x >= 0 else ""

    if "recall" in results_by_method and "query" in results_by_method:
        r_recall = results_by_method["recall"]
        r_query = results_by_method["query"]
        delta_r1 = (sum(r.hit_at_1 for r in r_recall) - sum(r.hit_at_1 for r in r_query)) / n
        delta_r3 = (sum(r.hit_at_3 for r in r_recall) - sum(r.hit_at_3 for r in r_query)) / n
        delta_lat = (sum(r.latency_ms for r in r_recall) - sum(r.latency_ms for r in r_query)) / n
        print(f"  /recall vs /query:")
        print(f"    Recall@1  {sign(delta_r1)}{delta_r1:+.1%}")
        print(f"    Recall@3  {sign(delta_r3)}{delta_r3:+.1%}")
        print(f"    Latency   {sign(delta_lat)}{delta_lat:+.0f}ms  (cost of filter call)")
        print()

    if "recall_b" in results_by_method and "query" in results_by_method:
        r_b = results_by_method["recall_b"]
        r_query = results_by_method["query"]
        delta_r1 = (sum(r.hit_at_1 for r in r_b) - sum(r.hit_at_1 for r in r_query)) / n
        delta_r3 = (sum(r.hit_at_3 for r in r_b) - sum(r.hit_at_3 for r in r_query)) / n
        delta_lat = (sum(r.latency_ms for r in r_b) - sum(r.latency_ms for r in r_query)) / n
        print(f"  /recall_b vs /query (zero-LLM decomposition baseline):")
        print(f"    Recall@1  {sign(delta_r1)}{delta_r1:+.1%}")
        print(f"    Recall@3  {sign(delta_r3)}{delta_r3:+.1%}")
        print(f"    Latency   {sign(delta_lat)}{delta_lat:+.0f}ms")
        print()


# ── main ───────────────────────────────────────────────────────────────────

def load_cases(path: Path) -> list[Case]:
    with open(path) as f:
        raw = json.load(f)
    return [Case(**c) for c in raw]


def main():
    parser = argparse.ArgumentParser(description="cogito benchmark")
    parser.add_argument("--cases", default=str(Path(__file__).parent / "cases.json"),
                        help="Path to test cases JSON")
    parser.add_argument("--port", type=int, default=int(os.environ.get("COGITO_PORT", "19420")))
    parser.add_argument("--limit", type=int, default=50, help="Candidate limit for /recall")
    parser.add_argument("--query-limit", type=int, default=5, help="Result limit for /query")
    parser.add_argument("--methods", default="recall,query,recall_b",
                        help="Comma-separated methods to test (recall, query, recall_b)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    methods = [m.strip() for m in args.methods.split(",")]
    cases_path = Path(args.cases)

    if not cases_path.exists():
        print(f"No test cases file at {cases_path}")
        print("Create bench/cases.json — see benchmark.py docstring for format.")
        sys.exit(1)

    cases = load_cases(cases_path)
    print(f"Loaded {len(cases)} test cases from {cases_path}")
    print(f"Running methods: {', '.join(methods)}")

    results_by_method: dict[str, list[Result]] = {}
    for method in methods:
        # recall and recall_b both use broad candidate pool; query uses narrow limit
        limit = args.limit if method in ("recall", "recall_b") else args.query_limit
        print(f"\nRunning /{method} (limit={limit})...")
        results = []
        for case in cases:
            r = run_case(base_url, case, method, limit)
            mark = "✅" if r.hit_at_1 else ("🟡" if r.hit_at_3 else "❌")
            print(f"  {mark} {case.query[:50]:<50}  {r.latency_ms:.0f}ms  [{r.api_method}]")
            results.append(r)
        results_by_method[method] = results

    print_results(results_by_method, cases, args.verbose)


if __name__ == "__main__":
    main()
