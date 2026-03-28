#!/usr/bin/env python3
"""
cogito-ergo combined-system eval.

Measures the four retrieval modes independently and combined:
  A  snapshot only      — snapshot.md text, no vector query
  B  snapshot + recall  — full combined system (the real thing)
  C  recall only        — /recall (two-stage + LLM filter), no snapshot
  D  recall_b only      — /recall_b (zero-LLM multi-query RRF), no snapshot

Four case types:
  direct_recall        — paraphrase queries for specific stored facts
                         (dynamic: generated from live corpus; queries use
                         word positions that avoid copying the memory's own
                         first words — so it's not a trivial prefix lookup)
  cross_reference      — multi-topic aggregation queries primarily answered
                         by the snapshot layer, not per-memory lookup
  semantic_gap         — vocabulary mismatch queries (vocab_map territory):
                         user phrasing diverges from corpus terminology
  adversarial_negative — near-miss off-topic queries; should return nothing
                         relevant (these look similar to real topics but the
                         specific fact isn't in the corpus)

Scoring:
  Positive cases:  keyword recall = fraction of expected[] keywords found
                   in the combined returned text. A case is a HIT if recall >= 0.25
                   (at least one keyword per four expected). MRR scores rank-1.
  Adversarial:     1.0 if no memories returned; 0.0 otherwise. Mode A
                   adversarial is scored as 0 (snapshot always returns text).

Output:
  Per-mode avg recall by type, overall recall@1, MRR, latency, token cost,
  win/tie/loss paired comparison (B vs A, B vs C, B vs D).

Usage:
  python bench/eval.py
  python bench/eval.py --port 19420 --verbose
  python bench/eval.py --static-only        # skip dynamic case generation
  python bench/eval.py --n 30               # how many dynamic cases to generate
  python bench/eval.py --modes A,B,C,D
  python bench/eval.py --cases bench/eval_cases.json   # static cases file
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


HIT_THRESHOLD = 0.25   # fraction of expected keywords required for a hit
SNAPSHOT_MAX_TOKENS = 1200   # chars/4 — approx token budget logged for mode A


# ── data types ─────────────────────────────────────────────────────────────

@dataclass
class Case:
    query: str
    expected: list[str]          # expected keywords (empty = adversarial)
    case_type: str               # direct_recall | cross_reference | semantic_gap | adversarial_negative
    notes: str = ""
    source_memory: str = ""      # for dynamic cases: the memory text this was derived from


@dataclass
class ModeResult:
    mode: str
    case: Case
    returned_text: str           # all returned memory texts joined, or snapshot text for A
    memories: list[dict]         # raw memories list (empty for mode A)
    latency_ms: float
    keyword_recall: float        # fraction of expected keywords found
    is_hit: bool                 # True if keyword_recall >= HIT_THRESHOLD (or adversarial correct)
    rank_1_hit: bool = False     # True if first returned memory contains a keyword
    rr: float = 0.0              # reciprocal rank


# ── server helpers ──────────────────────────────────────────────────────────

def _get(base_url: str, path: str, timeout: int = 10) -> dict:
    try:
        with urllib.request.urlopen(f"{base_url}{path}", timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"\nError: server not reachable at {base_url}{path} — {e}", file=sys.stderr)
        sys.exit(1)


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
        print(f"\nError: server not reachable at {base_url}{path} — {e}", file=sys.stderr)
        sys.exit(1)


# ── scoring ─────────────────────────────────────────────────────────────────

def _keyword_recall(text: str, expected: list[str]) -> float:
    """Fraction of expected keywords found (case-insensitive substring)."""
    if not expected:
        return 0.0
    t = text.lower()
    found = sum(1 for kw in expected if kw.lower() in t)
    return found / len(expected)


def _score_result(
    case: Case,
    memories: list[dict],
    snapshot_text: str,
    mode: str,
    latency_ms: float,
) -> ModeResult:
    """Score a single retrieval result for one mode."""

    # Mode A: snapshot text only, no memories
    if mode == "A":
        if case.case_type == "adversarial_negative":
            # Snapshot always returns text — adversarial is meaningless for mode A
            # Score as 0 (it can't filter)
            return ModeResult(
                mode=mode, case=case, returned_text=snapshot_text,
                memories=[], latency_ms=latency_ms,
                keyword_recall=0.0, is_hit=False,
            )
        recall = _keyword_recall(snapshot_text, case.expected)
        is_hit = recall >= HIT_THRESHOLD
        return ModeResult(
            mode=mode, case=case, returned_text=snapshot_text,
            memories=[], latency_ms=latency_ms,
            keyword_recall=round(recall, 3), is_hit=is_hit,
            rank_1_hit=is_hit,   # snapshot is flat, no rank
            rr=1.0 if is_hit else 0.0,
        )

    # Modes B, C, D: vector retrieval (+ snapshot for B)
    if case.case_type == "adversarial_negative":
        # Correct answer: nothing returned
        is_correct = len(memories) == 0
        return ModeResult(
            mode=mode, case=case,
            returned_text=" ".join(m.get("text", "") for m in memories),
            memories=memories, latency_ms=latency_ms,
            keyword_recall=1.0 if is_correct else 0.0,
            is_hit=is_correct,
            rank_1_hit=is_correct,
            rr=1.0 if is_correct else 0.0,
        )

    all_text_parts = []
    if mode == "B" and snapshot_text:
        all_text_parts.append(snapshot_text)

    mem_texts = [m.get("text", "") for m in memories]
    all_text_parts.extend(mem_texts)
    combined = "\n".join(all_text_parts)

    overall_recall = _keyword_recall(combined, case.expected)
    is_hit = overall_recall >= HIT_THRESHOLD

    # Rank-1 hit: first memory (not snapshot) contains a keyword
    rank_1_hit = bool(mem_texts) and _keyword_recall(mem_texts[0], case.expected) >= HIT_THRESHOLD
    rr = 0.0
    for i, mt in enumerate(mem_texts, 1):
        if _keyword_recall(mt, case.expected) >= HIT_THRESHOLD:
            rr = 1.0 / i
            break
    # Mode B: if snapshot alone answers it, count as rank-1
    if mode == "B" and not rr and snapshot_text:
        snap_recall = _keyword_recall(snapshot_text, case.expected)
        if snap_recall >= HIT_THRESHOLD:
            rank_1_hit = True
            rr = 1.0

    return ModeResult(
        mode=mode, case=case, returned_text=combined,
        memories=memories, latency_ms=latency_ms,
        keyword_recall=round(overall_recall, 3),
        is_hit=is_hit, rank_1_hit=rank_1_hit, rr=round(rr, 3),
    )


# ── dynamic case generation ─────────────────────────────────────────────────

def _extract_key_terms(text: str) -> list[str]:
    """
    Extract useful keywords from a memory text.
    Prefers: proper nouns (Title Case), version strings, technical tokens.
    Falls back to long lowercase words. Avoids first-word bias.
    """
    words = text.split()
    terms: list[str] = []

    # Proper nouns and version strings from anywhere in the text
    for w in words:
        clean = w.strip(".,;:\"'()")
        if not clean:
            continue
        # Version strings: v0.3, 1.0.5, etc.
        if re.match(r"v?\d+\.\d+", clean):
            terms.append(clean)
        # CamelCase or ALL_CAPS technical tokens
        elif re.match(r"[A-Z][a-z]+[A-Z]", clean) or re.match(r"[A-Z]{3,}", clean):
            terms.append(clean)
        # Title Case words (not at position 0 — avoids sentence-start bias)
        elif clean[0].isupper() and len(clean) > 3 and words.index(w) > 0:
            terms.append(clean)

    # Fallback: long words from middle/end of text
    if len(terms) < 2:
        mid = len(words) // 2
        for w in words[mid:]:
            clean = w.strip(".,;:\"'()")
            if len(clean) > 5:
                terms.append(clean)

    return list(dict.fromkeys(terms))[:5]  # dedupe, max 5


def _make_query_from_memory(text: str) -> str:
    """
    Build a natural query from a memory text WITHOUT using the first words.
    Uses the second half of the text as the basis, so it's not a trivial
    prefix lookup.
    """
    words = text.split()
    if len(words) < 6:
        return text  # too short to avoid first-word bias, use as-is

    # Use words from second half
    half = len(words) // 2
    pivot_words = words[half : half + 5]
    pivot = " ".join(pivot_words).strip(".,;:")

    templates = [
        f"what do you know about {pivot}",
        f"details on {pivot}",
        f"explain {pivot}",
        f"context for {pivot}",
    ]
    return templates[hash(text) % len(templates)]


def generate_dynamic_cases(base_url: str, n: int, rng: random.Random) -> list[Case]:
    """
    Generate direct_recall cases from live corpus memories.
    Fetches memories via /recall with a broad seed query, then generates
    paraphrase queries from mid/end of each memory (avoids first-word bias).
    """
    # Use a few seed queries to pull diverse memories
    seeds = [
        "project tool architecture",
        "bug fix incident error",
        "version release published",
        "agent memory retrieval",
    ]
    seen_texts: set[str] = set()
    memories: list[str] = []

    for seed in seeds:
        try:
            resp, _ = _post(base_url, "/recall_b", {"text": seed, "limit": 25}, timeout=15)
            for m in resp.get("memories", []):
                t = m.get("text", "").strip()
                if t and t not in seen_texts and len(t.split()) >= 5:
                    seen_texts.add(t)
                    memories.append(t)
        except Exception:
            pass

    if not memories:
        return []

    rng.shuffle(memories)
    selected = memories[: min(n, len(memories))]

    cases = []
    for mem in selected:
        query = _make_query_from_memory(mem)
        expected = _extract_key_terms(mem)
        if not expected:
            continue
        cases.append(Case(
            query=query,
            expected=expected,
            case_type="direct_recall",
            notes=f"dynamic — generated from corpus",
            source_memory=mem,
        ))

    return cases


# ── static case loading ─────────────────────────────────────────────────────

def load_static_cases(path: Path) -> list[Case]:
    if not path.exists():
        return []
    with open(path) as f:
        raw = json.load(f)
    return [Case(**c) for c in raw]


# ── eval runner ─────────────────────────────────────────────────────────────

def run_mode(
    mode: str,
    cases: list[Case],
    base_url: str,
    snapshot_text: str,
    limit: int,
) -> list[ModeResult]:
    results = []
    for case in cases:
        if mode == "A":
            # No server call — just snapshot text
            t0 = time.monotonic()
            latency = (time.monotonic() - t0) * 1000  # ~0ms
            r = _score_result(case, [], snapshot_text, "A", latency)

        elif mode == "B":
            resp, latency = _post(base_url, "/recall", {"text": case.query, "limit": limit})
            memories = resp.get("memories", [])
            r = _score_result(case, memories, snapshot_text, "B", latency)

        elif mode == "C":
            resp, latency = _post(base_url, "/recall", {"text": case.query, "limit": limit})
            memories = resp.get("memories", [])
            r = _score_result(case, memories, "", "C", latency)

        elif mode == "D":
            resp, latency = _post(base_url, "/recall_b", {"text": case.query, "limit": limit})
            memories = resp.get("memories", [])
            r = _score_result(case, memories, "", "D", latency)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        results.append(r)
    return results


# ── reporting ───────────────────────────────────────────────────────────────

CASE_TYPES = ["direct_recall", "cross_reference", "semantic_gap", "adversarial_negative"]
MODE_LABELS = {
    "A": "snapshot only",
    "B": "snapshot + recall",
    "C": "recall only",
    "D": "recall_b only",
}


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def print_report(
    results_by_mode: dict[str, list[ModeResult]],
    cases: list[Case],
    snapshot_text: str,
    verbose: bool,
) -> None:
    n = len(cases)
    modes = list(results_by_mode.keys())

    print(f"\n{'═' * 72}")
    print(f"  cogito combined-system eval  —  {n} cases")
    print(f"{'═' * 72}\n")

    # Per-case verbose output
    if verbose:
        for i, case in enumerate(cases):
            print(f"  [{i+1:02d}] [{case.case_type}]  {case.query}")
            if case.notes:
                print(f"        note: {case.notes}")
            for mode in modes:
                r = results_by_mode[mode][i]
                if case.case_type == "adversarial_negative":
                    mark = "✅" if r.is_hit else "❌"
                    detail = f"returned {len(r.memories)} memories"
                else:
                    mark = "✅" if r.rank_1_hit else ("🟡" if r.is_hit else "❌")
                    first = (r.memories[0]["text"][:55] if r.memories else
                             (r.returned_text[:55] if mode == "A" else "(empty)"))
                    detail = f"recall={r.keyword_recall:.0%}  {first!r}"
                print(f"        Mode {mode} {mark}  {r.latency_ms:5.0f}ms  {detail}")
            print()

    # ── Overall metrics table ──
    print(f"  {'Metric':<26}", end="")
    for m in modes:
        label = f"{m}:{MODE_LABELS[m][:14]}"
        print(f"  {label:>18}", end="")
    print()
    print(f"  {'─' * 26}", end="")
    for _ in modes:
        print(f"  {'─' * 18}", end="")
    print()

    def row(label: str, fn):
        print(f"  {label:<26}", end="")
        for mode in modes:
            rs = results_by_mode[mode]
            val = fn(rs)
            if isinstance(val, float) and val <= 1.0 and "ms" not in label:
                print(f"  {val:>17.1%} ", end="")
            else:
                print(f"  {val:>17.0f} ", end="")
        print()

    # Exclude adversarial from recall metrics
    def pos_results(rs: list[ModeResult]) -> list[ModeResult]:
        return [r for r in rs if r.case.case_type != "adversarial_negative"]

    pos_n = sum(1 for c in cases if c.case_type != "adversarial_negative")
    adv_n = n - pos_n

    row("Recall@1 (pos cases)", lambda rs: _avg([r.rank_1_hit for r in pos_results(rs)]))
    row("Hit@any (pos cases)", lambda rs: _avg([r.is_hit for r in pos_results(rs)]))
    row("MRR (pos cases)", lambda rs: _avg([r.rr for r in pos_results(rs)]))
    if adv_n > 0:
        row("Adversarial precision", lambda rs: _avg(
            [r.is_hit for r in rs if r.case.case_type == "adversarial_negative"]
        ))
    row("Avg latency ms", lambda rs: _avg([r.latency_ms for r in rs]))
    snap_tok = len(snapshot_text) // 4
    print(f"  {'Snapshot tokens (A/B)':<26}", end="")
    for mode in modes:
        tok = snap_tok if mode in ("A", "B") else 0
        print(f"  {tok:>17} ", end="")
    print()

    # ── By case type ──
    print(f"\n  {'─' * 72}")
    print("  By case type (Recall@1 / Hit@any):\n")
    for ct in CASE_TYPES:
        ct_cases = [c for c in cases if c.case_type == ct]
        if not ct_cases:
            continue
        ct_n = len(ct_cases)
        print(f"  {ct} (n={ct_n}):")
        for mode in modes:
            rs = [results_by_mode[mode][i] for i, c in enumerate(cases) if c.case_type == ct]
            if ct == "adversarial_negative":
                adv_acc = _avg([r.is_hit for r in rs])
                print(f"    Mode {mode}: precision={adv_acc:.0%}  (correct if 0 results returned)")
            else:
                r1 = _avg([r.rank_1_hit for r in rs])
                hit = _avg([r.is_hit for r in rs])
                print(f"    Mode {mode}: R@1={r1:.0%}  hit@any={hit:.0%}")
        print()

    # ── Paired comparisons ──
    if "B" in results_by_mode:
        print(f"  {'─' * 72}")
        print("  Paired comparison (combined system = Mode B):\n")
        for other in [m for m in modes if m != "B"]:
            rs_b = [r for r in results_by_mode["B"] if r.case.case_type != "adversarial_negative"]
            rs_o = [r for r in results_by_mode[other] if r.case.case_type != "adversarial_negative"]
            wins = sum(1 for b, o in zip(rs_b, rs_o) if b.is_hit and not o.is_hit)
            ties = sum(1 for b, o in zip(rs_b, rs_o) if b.is_hit == o.is_hit)
            losses = sum(1 for b, o in zip(rs_b, rs_o) if not b.is_hit and o.is_hit)
            print(f"    B vs {other} ({MODE_LABELS.get(other, other)}): "
                  f"{wins} wins / {ties} ties / {losses} losses")
        print()

    # ── Verdict ──
    if len(modes) >= 2:
        print(f"  {'─' * 72}")
        b_score = _avg([r.is_hit for r in results_by_mode.get("B", [])
                        if r.case.case_type != "adversarial_negative"]) if "B" in results_by_mode else 0
        c_score = _avg([r.is_hit for r in results_by_mode.get("C", [])
                        if r.case.case_type != "adversarial_negative"]) if "C" in results_by_mode else 0
        a_score = _avg([r.is_hit for r in results_by_mode.get("A", [])
                        if r.case.case_type != "adversarial_negative"]) if "A" in results_by_mode else 0

        print(f"\n  VERDICT:")
        if "B" in results_by_mode and "C" in results_by_mode and "A" in results_by_mode:
            snap_delta = b_score - c_score
            retrieval_delta = b_score - a_score
            print(f"    Snapshot contributes: {snap_delta:+.0%} vs recall-only (B vs C)")
            print(f"    Retrieval contributes: {retrieval_delta:+.0%} vs snapshot-only (B vs A)")
            if b_score >= c_score and b_score >= a_score:
                print(f"    Combined system (B) is best at {b_score:.0%} hit@any")
            elif c_score > b_score:
                print(f"    Recall-only (C) beats combined: {c_score:.0%} vs {b_score:.0%}")
                print(f"    Snapshot may be adding noise or the snapshot.md needs rebuild")
            else:
                print(f"    Snapshot-only (A) beats combined: {a_score:.0%} vs {b_score:.0%}")
                print(f"    Vector retrieval may be adding noise")
        print()

    print(f"{'═' * 72}\n")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="cogito combined-system eval")
    parser.add_argument("--port", type=int, default=int(os.environ.get("COGITO_PORT", "19420")))
    parser.add_argument("--cases", default=str(Path(__file__).parent / "eval_cases.json"),
                        help="Static cases file (cross_reference, semantic_gap, adversarial)")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of dynamic direct_recall cases to generate (default: 20)")
    parser.add_argument("--static-only", action="store_true",
                        help="Skip dynamic case generation")
    parser.add_argument("--modes", default="A,B,C,D",
                        help="Comma-separated modes to run (default: A,B,C,D)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Candidate limit for /recall and /recall_b (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for dynamic cases")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    modes = [m.strip() for m in args.modes.split(",")]

    # Load snapshot
    print(f"[eval] Fetching snapshot from {base_url}/snapshot...")
    snap_resp = _get(base_url, "/snapshot", timeout=5)
    if "error" in snap_resp:
        print(f"[eval] No snapshot: {snap_resp['error']}")
        print("[eval] Run `cogito snapshot` first, or use --modes C,D to skip snapshot modes.")
        if any(m in modes for m in ("A", "B")):
            modes = [m for m in modes if m not in ("A", "B")]
            print(f"[eval] Continuing with modes: {modes}")
    snapshot_text = snap_resp.get("snapshot", "")
    print(f"[eval] Snapshot: {len(snapshot_text)} chars (~{len(snapshot_text)//4} tokens)")

    # Load static cases
    static_path = Path(args.cases)
    static_cases = load_static_cases(static_path)
    print(f"[eval] Static cases: {len(static_cases)} from {static_path}")

    # Generate dynamic cases
    dynamic_cases: list[Case] = []
    if not args.static_only and args.n > 0:
        print(f"[eval] Generating {args.n} dynamic direct_recall cases from corpus...")
        rng = random.Random(args.seed)
        dynamic_cases = generate_dynamic_cases(base_url, args.n, rng)
        print(f"[eval] Generated {len(dynamic_cases)} dynamic cases")

    all_cases = static_cases + dynamic_cases
    if not all_cases:
        print("[eval] No cases to run. Add bench/eval_cases.json or use --n > 0.")
        sys.exit(1)

    by_type = {ct: sum(1 for c in all_cases if c.case_type == ct) for ct in CASE_TYPES}
    print(f"[eval] Total cases: {len(all_cases)}")
    for ct, cnt in by_type.items():
        if cnt:
            print(f"         {ct}: {cnt}")
    print(f"[eval] Running modes: {', '.join(f'{m} ({MODE_LABELS.get(m, m)})' for m in modes)}\n")

    results_by_mode: dict[str, list[ModeResult]] = {}
    for mode in modes:
        print(f"[eval] Running mode {mode} ({MODE_LABELS.get(mode, mode)})...")
        results = run_mode(mode, all_cases, base_url, snapshot_text, args.limit)
        results_by_mode[mode] = results
        pos = [r for r in results if r.case.case_type != "adversarial_negative"]
        r1 = _avg([r.rank_1_hit for r in pos]) if pos else 0
        hit = _avg([r.is_hit for r in pos]) if pos else 0
        lat = _avg([r.latency_ms for r in results])
        print(f"         R@1={r1:.0%}  hit@any={hit:.0%}  avg_lat={lat:.0f}ms")

    print_report(results_by_mode, all_cases, snapshot_text, args.verbose)


if __name__ == "__main__":
    main()
