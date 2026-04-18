"""
Test: COMBINED (BM25 + turn-level chunking + nomic prefixes).

Combines all improvements:
1. Turn-level chunking (from longmemeval_turnlevel.py): sessions split into
   overlapping turn-pair chunks before embedding.
2. BM25 + dense hybrid retrieval (from longmemeval_hybrid.py): both cosine
   and BM25 searches run per sub-query, all ranked lists fused via RRF.
3. Nomic search_query/search_document prefixes on all embeds.

After chunk-level retrieval, chunks are mapped back to sessions via
session_id dedup for final session-level scoring.
"""

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

import bm25s

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STOP = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall",
    "what", "which", "who", "whom", "whose", "how", "why", "when", "where",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "this", "that", "these", "those", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "into", "through", "during", "before",
    "after", "above", "below", "and", "or", "but", "if", "while", "because",
    "so", "not", "no", "nor", "yet", "than", "then", "about",
    "cause", "caused", "describe", "explain", "tell", "give", "show",
    "find", "get", "make", "use", "used", "using", "their", "its",
    "can", "cannot",
})
MAX_SUBQUERIES = 8
_RRF_K = 60
_COSINE_WEIGHT = 0.7
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

# Nomic prefixes
QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "


# ---------------------------------------------------------------------------
# Chunking (copied exactly from longmemeval_turnlevel.py)
# ---------------------------------------------------------------------------
def chunk_session(session: list[dict], session_id: str) -> list[tuple[str, str]]:
    """
    Chunk a session into turn-level units.
    Each chunk = previous turn pair (context) + current turn pair.
    Returns list of (chunk_text, session_id).
    """
    chunks = []
    turns = []  # collect (user_msg, assistant_msg) pairs

    current_user = None
    for msg in session:
        if msg["role"] == "user":
            current_user = msg["content"]
        elif msg["role"] == "assistant" and current_user is not None:
            turns.append((current_user, msg["content"]))
            current_user = None

    # If there are unpaired user messages, add them as solo turns
    if current_user is not None:
        turns.append((current_user, ""))

    # If no turns extracted, use the raw session text
    if not turns:
        raw = " ".join(m["content"] for m in session if m["role"] == "user")
        if raw.strip():
            chunks.append((raw, session_id))
        return chunks

    for i, (user, assistant) in enumerate(turns):
        parts = []
        # Previous turn as context (1-turn overlap)
        if i > 0:
            prev_u, prev_a = turns[i - 1]
            parts.append(f"[context] {prev_u}")
            if prev_a:
                parts.append(f"[context] {prev_a}")
        # Current turn
        parts.append(user)
        if assistant:
            parts.append(assistant)

        chunk_text = " ".join(parts)
        chunks.append((chunk_text, session_id))

    return chunks


def dedup_to_sessions(
    ranked_chunk_indices: list[int],
    chunk_session_ids: list[str],
    session_id_to_corpus_idx: dict[str, int],
) -> list[int]:
    """
    Map ranked chunk indices back to session (corpus) indices.
    First occurrence of each session_id determines that session's rank.
    Returns deduplicated list of corpus indices in ranked order.
    """
    seen_sids: set[str] = set()
    ranked_sessions: list[int] = []
    for chunk_idx in ranked_chunk_indices:
        sid = chunk_session_ids[chunk_idx]
        if sid not in seen_sids:
            seen_sids.add(sid)
            corpus_idx = session_id_to_corpus_idx[sid]
            ranked_sessions.append(corpus_idx)
    # Append any sessions not reached via chunks (shouldn't happen, but be safe)
    all_corpus_indices = set(session_id_to_corpus_idx.values())
    for idx in sorted(all_corpus_indices - set(ranked_sessions)):
        ranked_sessions.append(idx)
    return ranked_sessions


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def batch_embed(texts: list[str], retries: int = 8) -> list[list[float]] | None:
    sanitized = [t[:2000] if len(t) > 2000 else t if t.strip() else "empty" for t in texts]
    for attempt in range(retries):
        try:
            body = json.dumps({"model": EMBED_MODEL, "input": sanitized}).encode()
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/embed", data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            vecs = data.get("embeddings", [])
            return vecs if len(vecs) == len(texts) else None
        except Exception as e:
            if attempt < retries - 1:
                wait = min(2 ** attempt, 8)
                print(f"  [embed retry {attempt+1}/{retries}] {e} — waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  [embed FAILED] {e}", file=sys.stderr)
                return None


def batch_embed_docs(texts: list[str]) -> list[list[float]] | None:
    """Embed documents with search_document: prefix."""
    prefixed = [DOC_PREFIX + t for t in texts]
    return batch_embed(prefixed)


def batch_embed_queries(texts: list[str]) -> list[list[float]] | None:
    """Embed queries with search_query: prefix."""
    prefixed = [QUERY_PREFIX + t for t in texts]
    return batch_embed(prefixed)


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Query decomposition
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s-]", " ", text.lower()).split()


def key_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOP and len(t) >= 2]


def build_subqueries(query: str) -> list[str]:
    seen, result = set(), []
    def add(q):
        q = q.strip()
        if q and q not in seen:
            seen.add(q); result.append(q)
    add(query)
    tokens = key_tokens(query)
    if not tokens:
        return result
    add(" ".join(tokens))
    for i in range(len(tokens) - 1):
        add(f"{tokens[i]} {tokens[i+1]}")
    for i in range(len(tokens) - 2):
        add(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
    for t in tokens:
        if len(t) >= 3:
            add(t)
    return result[:MAX_SUBQUERIES]


# ---------------------------------------------------------------------------
# Retrieval: sub-query decomposition + cosine + BM25 → RRF fusion
# (operates on chunks; BM25 logic copied from longmemeval_hybrid.py)
# ---------------------------------------------------------------------------
def retrieve_chunks(
    query: str,
    chunk_texts: list[str],
    chunk_vecs: list[list[float]],
    bm25_index: bm25s.BM25,
) -> list[int]:
    """Rank chunk indices by relevance via BM25 + dense cosine, fused with RRF."""
    subqueries = build_subqueries(query)

    # Embed subqueries with search_query: prefix
    sq_vecs = batch_embed_queries(subqueries)
    if sq_vecs is None:
        sq_vecs = batch_embed_queries([query])
        if sq_vecs is None:
            return list(range(len(chunk_texts)))

    k_bm25 = min(20, len(chunk_texts))
    runs: list[list[tuple[int, float]]] = []

    for subquery, sv in zip(subqueries, sq_vecs):
        # --- Dense cosine run ---
        scored = [(i, cosine_sim(sv, cv)) for i, cv in enumerate(chunk_vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        runs.append(scored[:20])

        # --- BM25 run ---
        query_tokens = bm25s.tokenize([subquery])
        bm25_results, bm25_scores = bm25_index.retrieve(query_tokens, k=k_bm25)
        # bm25_results[0] and bm25_scores[0] are numpy arrays of doc indices / scores
        bm25_run: list[tuple[int, float]] = [
            (int(idx), float(score))
            for idx, score in zip(bm25_results[0], bm25_scores[0])
        ]
        runs.append(bm25_run)

    # RRF merge over all runs (cosine + BM25 interleaved)
    rrf_scores: dict[int, float] = {}
    for run in runs:
        for rank, (idx, _score) in enumerate(run, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank)

    query_vec = sq_vecs[0]
    cosine_scores = {i: cosine_sim(query_vec, chunk_vecs[i]) for i in rrf_scores}
    rrf_max = max(rrf_scores.values()) if rrf_scores else 1.0

    blended: list[tuple[int, float]] = []
    for idx in rrf_scores:
        rrf_n = rrf_scores[idx] / rrf_max if rrf_max > 0 else 0.0
        cos = cosine_scores.get(idx, 0.0)
        score = (1.0 - _COSINE_WEIGHT) * rrf_n + _COSINE_WEIGHT * cos
        blended.append((idx, score))

    blended.sort(key=lambda x: x[1], reverse=True)

    ranked_ids = [idx for idx, _ in blended]
    remaining = [i for i in range(len(chunk_texts)) if i not in rrf_scores]
    remaining_scored = [(i, cosine_sim(query_vec, chunk_vecs[i])) for i in remaining]
    remaining_scored.sort(key=lambda x: x[1], reverse=True)
    ranked_ids.extend([i for i, _ in remaining_scored])

    return ranked_ids


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def dcg(relevances, k):
    import numpy as np
    relevances = np.asarray(relevances, dtype=float)[:k]
    if relevances.size:
        return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))
    return 0.0


def ndcg_score(rankings, correct_indices, n_corpus, k):
    import numpy as np
    relevances = [1 if i in correct_indices else 0 for i in range(n_corpus)]
    sorted_rel = [relevances[idx] for idx in rankings[:k]]
    ideal_rel = sorted(relevances, reverse=True)
    ideal = dcg(ideal_rel, k)
    actual = dcg(sorted_rel, k)
    return actual / ideal if ideal > 0 else 0.0


def evaluate_retrieval(rankings, correct_indices, k):
    recalled = set(rankings[:k])
    recall_any = float(any(idx in recalled for idx in correct_indices))
    recall_all = float(all(idx in recalled for idx in correct_indices))
    return recall_any, recall_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["s", "m"], default="s")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data_dir", default=str(Path(__file__).resolve().parent.parent.parent / "LongMemEval" / "data"))
    args = parser.parse_args()

    split_file = f"longmemeval_{args.split}_cleaned.json"
    data_path = Path(args.data_dir) / split_file
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    print(f"COMBINED (BM25 + turn-level + prefixes)")
    print(f"Loading {data_path.name}...")
    data = json.load(open(data_path))
    data = [e for e in data if "_abs" not in e["question_id"]]
    if args.limit > 0:
        data = data[:args.limit]
    print(f"[combined] Running on {len(data)} questions — BM25 + turn-level chunks + nomic prefixes")

    metrics = {k: {"recall_any": [], "recall_all": [], "ndcg": []} for k in [1, 3, 5, 10]}
    total_time = 0.0

    for qi, entry in enumerate(data):
        question = entry["question"]
        answer_sids = set(entry["answer_session_ids"])

        # --- Build session corpus (for scoring reference) ---
        corpus_ids: list[str] = []
        correct_indices: list[int] = []
        session_id_to_corpus_idx: dict[str, int] = {}

        for sid, session in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
            corpus_idx = len(corpus_ids)
            corpus_ids.append(sid)
            session_id_to_corpus_idx[sid] = corpus_idx
            if sid in answer_sids:
                correct_indices.append(corpus_idx)

        if not correct_indices:
            continue

        # --- Build chunk corpus ---
        all_chunks: list[tuple[str, str]] = []  # (chunk_text, session_id)
        for sid, session in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
            session_chunks = chunk_session(session, sid)
            all_chunks.extend(session_chunks)

        chunk_texts = [c[0] for c in all_chunks]
        chunk_session_ids = [c[1] for c in all_chunks]
        n_sessions = len(corpus_ids)
        n_chunks = len(chunk_texts)

        # Validation print for first 5 questions
        if qi < 5:
            avg_chunks = n_chunks / n_sessions if n_sessions else 0
            print(f"  q{qi+1}: {n_sessions} sessions → {n_chunks} chunks (avg {avg_chunks:.1f} chunks/session)")

        # --- Build per-question BM25 index over chunks ---
        corpus_tokens = bm25s.tokenize(chunk_texts)
        bm25_index = bm25s.BM25()
        bm25_index.index(corpus_tokens)

        # --- Embed chunks with search_document: prefix ---
        all_vecs: list[list[float]] = []
        embed_ok = True
        for i in range(0, len(chunk_texts), 100):
            batch = chunk_texts[i:i + 100]
            vecs = batch_embed_docs(batch)
            if vecs is None:
                print(f"  [{qi+1}] EMBED FAILED for chunk batch, skipping")
                embed_ok = False
                break
            all_vecs.extend(vecs)

        if not embed_ok:
            continue

        # --- Retrieve chunks (BM25 + dense → RRF), then map back to sessions ---
        t0 = time.time()
        ranked_chunk_indices = retrieve_chunks(question, chunk_texts, all_vecs, bm25_index)
        elapsed = time.time() - t0
        total_time += elapsed

        # Dedup: map chunk ranking → session ranking
        ranked_sessions = dedup_to_sessions(ranked_chunk_indices, chunk_session_ids, session_id_to_corpus_idx)

        # --- Score using session rankings ---
        for k in [1, 3, 5, 10]:
            r_any, r_all = evaluate_retrieval(ranked_sessions, correct_indices, k)
            n = ndcg_score(ranked_sessions, set(correct_indices), n_sessions, k)
            metrics[k]["recall_any"].append(r_any)
            metrics[k]["recall_all"].append(r_all)
            metrics[k]["ndcg"].append(n)

        r1_last = metrics[1]["recall_any"][-1]
        r5_last = metrics[5]["recall_any"][-1]
        status = "OK" if r5_last > 0 else "--"
        if (qi + 1) % 10 == 0 or qi == 0:
            running_r1 = sum(metrics[1]["recall_any"]) / len(metrics[1]["recall_any"])
            running_r5 = sum(metrics[5]["recall_any"]) / len(metrics[5]["recall_any"])
            print(
                f"  [{qi+1}/{len(data)}] R@1={running_r1:.1%} R@5={running_r5:.1%}  last={status}"
                f"  {elapsed:.1f}s  q={question[:50]}"
            )

    n = len(metrics[5]["recall_any"])
    print(f"\n{'='*70}")
    print(f"  COMBINED (BM25 + turn-level + prefixes): LongMemEval_{args.split.upper()}  —  {n} questions")
    print(f"{'='*70}\n")

    print(f"  {'Metric':<20} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    results = {}
    for row_name, key in [("recall_any", "recall_any"), ("recall_all", "recall_all"), ("ndcg", "ndcg")]:
        vals = []
        for k in [1, 3, 5, 10]:
            arr = metrics[k][key]
            vals.append(sum(arr) / len(arr) if arr else 0.0)
        results[row_name] = {1: vals[0], 3: vals[1], 5: vals[2], 10: vals[3]}
        print(f"  {row_name:<20} {vals[0]:>7.1%} {vals[1]:>7.1%} {vals[2]:>7.1%} {vals[3]:>7.1%}")

    avg_time = total_time / n if n else 0
    print(f"\n  Avg retrieval time: {avg_time*1000:.0f}ms")
    print(f"  Total time: {total_time:.1f}s")

    # Save results
    out = {
        "benchmark": "LongMemEval",
        "split": args.split.upper(),
        "date": "2026-04-15",
        "test": "combined_all",
        "engine": "BM25 (bm25s) + nomic-embed-text cosine on turn-level chunks, RRF fusion",
        "embed_model": "nomic-embed-text (Ollama)",
        "changes": (
            "Sessions chunked into turn-level units (prev turn as context + current turn). "
            "BM25 index built per-question over chunks (not whole sessions). "
            "BM25 and dense cosine both run per sub-query; all ranked lists fused via RRF. "
            "Chunk rankings deduped back to session rankings for scoring. "
            "search_query/search_document prefixes. 2000-char truncation."
        ),
        "questions_evaluated": n,
        "metrics": {
            "recall_any": {f"R@{k}": round(results["recall_any"][k], 3) for k in [1, 3, 5, 10]},
            "recall_all": {f"R@{k}": round(results["recall_all"][k], 3) for k in [1, 3, 5, 10]},
            "ndcg": {f"R@{k}": round(results["ndcg"][k], 3) for k in [1, 3, 5, 10]},
        },
        "avg_retrieval_time_ms": round(avg_time * 1000),
        "total_time_s": round(total_time, 1),
    }
    out_path = Path(__file__).parent / "results-combined-2026-04-15.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
