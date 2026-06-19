"""
bench/claude_code_session_eval.py — Real-world session retrieval eval.

Synthesizes 5 queries from ingested session titles/content and measures
top-1 hit rate against the known session.

Usage:
    cd /path/to/fidelis
    python3 bench/claude_code_session_eval.py

Requires: cogito-ergo installed (pip install -e .), Ollama running with nomic-embed-text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import chromadb
from cogito.recall_sessions import query_sessions

STORE_PATH = Path.home() / ".cogito" / "store"


def _get_sessions():
    """Load all session memories from ChromaDB."""
    client = chromadb.PersistentClient(path=str(STORE_PATH))
    col = client.get_collection("cogito_main")
    result = col.get(where={"mem_type": "session"}, include=["metadatas", "documents"])
    sessions = []
    for i, meta in enumerate(result["metadatas"]):
        sessions.append({
            "session_id": meta.get("session_id", ""),
            "turns_json": meta.get("turns_json", "[]"),
            "project_path": meta.get("project_path", ""),
            "start_ts": meta.get("start_ts", ""),
            "turn_count": meta.get("turn_count", 0),
        })
    return sessions


def _synthesize_queries(sessions) -> list[tuple[str, str]]:
    """
    Synthesize (query, expected_session_id) pairs from ingested sessions.
    Picks the 5 most turn-rich sessions and extracts keyword queries from
    the first user message of each.
    """
    # Sort by turn count descending — richest sessions give most signal
    ranked = sorted(sessions, key=lambda s: s["turn_count"], reverse=True)[:5]
    pairs = []
    for sess in ranked:
        try:
            turns = json.loads(sess["turns_json"])
        except Exception:
            turns = []
        # Use the first substantive user message as query seed
        user_messages = [t["content"] for t in turns if t.get("role") == "user" and len(t.get("content", "")) > 20]
        if not user_messages:
            continue
        # Take the first user message, extract up to 10 words as query
        seed = user_messages[0]
        query = " ".join(seed.split()[:10])
        pairs.append((query, sess["session_id"]))
    return pairs


def run_eval():
    print("=== cogito-ergo Claude Code Session Retrieval Eval ===\n")
    try:
        sessions = _get_sessions()
    except Exception as e:
        print(f"ERROR: Could not load sessions from store: {e}")
        print("Run: python3 -m cogito.ingest_claude_sessions")
        return None

    if len(sessions) < 2:
        print("ERROR: Fewer than 2 session memories ingested. Run ingest first.")
        return None

    print(f"Loaded {len(sessions)} session memories from store.\n")

    query_pairs = _synthesize_queries(sessions)
    if not query_pairs:
        print("ERROR: Could not synthesize queries — sessions may lack user turns.")
        return None

    print(f"Synthesized {len(query_pairs)} queries from top-{len(query_pairs)} richest sessions.")
    print("Scoring top-1 hit rate (did the correct session come back at rank 1?):\n")

    hits = 0
    results = []
    for i, (query, expected_id) in enumerate(query_pairs, 1):
        try:
            session_results = query_sessions(query, top_k=3)
        except Exception as e:
            print(f"  [{i}] QUERY ERROR: {e}")
            results.append({"query": query, "expected": expected_id, "hit": False, "error": str(e)})
            continue

        top_id = session_results[0].session_id if session_results else ""
        hit = top_id == expected_id
        hits += hit

        top_score = round(session_results[0].score, 3) if session_results else 0
        status = "HIT " if hit else "MISS"
        print(f"  [{i}] {status} | query={query[:60]!r}")
        print(f"        expected={expected_id[:8]} | got={top_id[:8]} | score={top_score}")
        if session_results and not hit:
            print(f"        top3 ids: {[r.session_id[:8] for r in session_results]}")
        results.append({
            "query": query,
            "expected": expected_id,
            "top_returned": top_id,
            "score": top_score,
            "hit": hit,
        })

    n = len(query_pairs)
    r1 = hits / n if n > 0 else 0.0
    print(f"\n=== RESULTS ===")
    print(f"  Top-1 hit rate: {hits}/{n} = {r1:.1%}")
    print(f"  Queries sourced from first user message of richest sessions (no external ground truth).")
    print()

    if r1 >= 0.70:
        print(f"PASS (>= 70% threshold). Publishable as real-world session recall number.")
        print(f"  Launch copy: '{r1:.0%} top-1 hit rate on {n}-query self-eval against {len(sessions)} ingested Claude Code sessions'")
    else:
        print(f"HOLD (<70%). Session claim should not appear in launch copy without this number or a better eval.")
        print(f"  Flag: needs 10-question user-validated eval before launch copy finalized.")

    print()
    print("NOTE: This eval uses synthetic queries derived from the stored sessions' own")
    print("text, making it an upper-bound estimate. A user-validated eval with independent")
    print("questions is required for a defensible production claim. This establishes the")
    print("plausibility floor.")
    return r1


if __name__ == "__main__":
    run_eval()
