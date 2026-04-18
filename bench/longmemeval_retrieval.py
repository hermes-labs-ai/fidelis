"""
Run cogito-ergo's recall_b retrieval against LongMemEval benchmark.

For each question:
  1. Build a corpus from the haystack sessions (session-level granularity)
  2. Embed all sessions + the query via nomic-embed-text (Ollama)
  3. Apply recall_b-style sub-query decomposition + RRF + cosine rerank
  4. Output ranked results in LongMemEval's expected format
  5. Compute retrieval metrics (recall@k, ndcg@k)

Usage:
  python bench/longmemeval_retrieval.py --split s --limit 50
  python bench/longmemeval_retrieval.py --split m --limit 50
"""

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (same as recall_b.py)
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


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def batch_embed(texts: list[str], retries: int = 8) -> list[list[float]] | None:
    """Batch-embed via Ollama. Returns list of vectors or None."""
    # nomic-embed-text effective context = 2048 tokens (nomic-bert.context_length).
    # Worst case (hex/code): ~2 chars/token → 2000 char safe limit.
    sanitized = [t[:2000] if len(t) > 2000 else t if t.strip() else "empty" for t in texts]
    for attempt in range(retries):
        try:
            body = json.dumps({
                "model": EMBED_MODEL,
                "input": sanitized,
                "options": {"num_ctx": 8192},
            }).encode()
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


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Query decomposition (from recall_b.py)
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
# Retrieval: sub-query decomposition + cosine rerank
# ---------------------------------------------------------------------------
def retrieve_sessions(query: str, corpus_texts: list[str], corpus_vecs: list[list[float]]) -> list[int]:
    """
    Given a query, corpus texts, and pre-computed corpus embeddings,
    return ranked indices into corpus (best first).
    """
    subqueries = build_subqueries(query)

    # Embed all subqueries in one batch
    sq_vecs = batch_embed(subqueries)
    if sq_vecs is None:
        # Fallback: embed just the original query
        sq_vecs = batch_embed([query])
        if sq_vecs is None:
            return list(range(len(corpus_texts)))  # give up, return original order

    # Run each sub-query against corpus via cosine similarity, get ranked lists
    runs: list[list[tuple[int, float]]] = []
    for sv in sq_vecs:
        scored = [(i, cosine_sim(sv, cv)) for i, cv in enumerate(corpus_vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        runs.append(scored[:20])  # top 20 per sub-query

    # RRF merge
    rrf_scores: dict[int, float] = {}
    for run in runs:
        for rank, (idx, _score) in enumerate(run, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank)

    # Cosine rerank against original query
    query_vec = sq_vecs[0]  # first sub-query IS the original query
    cosine_scores = {i: cosine_sim(query_vec, corpus_vecs[i]) for i in rrf_scores}

    # Normalize RRF
    rrf_max = max(rrf_scores.values()) if rrf_scores else 1.0

    # Blend
    blended: list[tuple[int, float]] = []
    for idx in rrf_scores:
        rrf_n = rrf_scores[idx] / rrf_max if rrf_max > 0 else 0.0
        cos = cosine_scores.get(idx, 0.0)
        score = (1.0 - _COSINE_WEIGHT) * rrf_n + _COSINE_WEIGHT * cos
        blended.append((idx, score))

    blended.sort(key=lambda x: x[1], reverse=True)

    # Return full ranking: blended candidates first, then remaining by cosine
    ranked_ids = [idx for idx, _ in blended]
    remaining = [i for i in range(len(corpus_texts)) if i not in rrf_scores]
    remaining_scored = [(i, cosine_sim(query_vec, corpus_vecs[i])) for i in remaining]
    remaining_scored.sort(key=lambda x: x[1], reverse=True)
    ranked_ids.extend([i for i, _ in remaining_scored])

    return ranked_ids


# ---------------------------------------------------------------------------
# Evaluation metrics (from LongMemEval eval_utils.py)
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
    parser.add_argument("--split", choices=["s", "m"], default="s", help="s=small (~50 sessions), m=medium (~500 sessions)")
    parser.add_argument("--limit", type=int, default=0, help="Limit to first N questions (0=all)")
    parser.add_argument("--data_dir", default=str(Path(__file__).resolve().parent.parent.parent / "LongMemEval" / "data"))
    args = parser.parse_args()

    split_file = f"longmemeval_{args.split}_cleaned.json"
    data_path = Path(args.data_dir) / split_file
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading {data_path.name}...")
    data = json.load(open(data_path))

    # Filter out abstention questions (no retrieval target)
    data = [e for e in data if "_abs" not in e["question_id"]]
    if args.limit > 0:
        data = data[:args.limit]
    print(f"Running on {len(data)} questions (non-abstention)")

    # Metrics accumulators
    metrics = {k: {"recall_any": [], "recall_all": [], "ndcg": []} for k in [1, 3, 5, 10]}
    total_time = 0.0

    for qi, entry in enumerate(data):
        qid = entry["question_id"]
        question = entry["question"]
        answer_sids = set(entry["answer_session_ids"])

        # Build corpus: one text per session (user turns only)
        corpus_texts = []
        corpus_ids = []
        correct_indices = []

        for sid, session in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
            text = " ".join(t["content"] for t in session if t["role"] == "user")
            corpus_texts.append(text)
            corpus_ids.append(sid)
            if sid in answer_sids:
                correct_indices.append(len(corpus_texts) - 1)

        if not correct_indices:
            continue

        # Pre-embed the corpus (one batch call)
        # For large corpora, chunk into batches of 100
        all_vecs = []
        for i in range(0, len(corpus_texts), 100):
            chunk = corpus_texts[i:i+100]
            vecs = batch_embed(chunk)
            if vecs is None:
                print(f"  [{qi+1}] EMBED FAILED for corpus chunk, skipping")
                break
            all_vecs.extend(vecs)
        else:
            # Run retrieval
            t0 = time.time()
            rankings = retrieve_sessions(question, corpus_texts, all_vecs)
            elapsed = time.time() - t0
            total_time += elapsed

            # Evaluate
            for k in [1, 3, 5, 10]:
                r_any, r_all = evaluate_retrieval(rankings, correct_indices, k)
                n = ndcg_score(rankings, set(correct_indices), len(corpus_texts), k)
                metrics[k]["recall_any"].append(r_any)
                metrics[k]["recall_all"].append(r_all)
                metrics[k]["ndcg"].append(n)

            # Progress
            r5 = metrics[5]["recall_any"][-1]
            status = "✅" if r5 > 0 else "❌"
            if (qi + 1) % 10 == 0 or qi == 0:
                running_r5 = sum(metrics[5]["recall_any"]) / len(metrics[5]["recall_any"])
                print(f"  [{qi+1}/{len(data)}] running R@5={running_r5:.1%}  last={status}  {elapsed:.1f}s  q={question[:50]}")
            continue
        # If embed failed, skip this question
        continue

    # Final results
    n = len(metrics[5]["recall_any"])
    print(f"\n{'='*70}")
    print(f"  cogito recall_b on LongMemEval_{args.split.upper()}  —  {n} questions")
    print(f"{'='*70}\n")

    print(f"  {'Metric':<20} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for row_name, key in [("recall_any", "recall_any"), ("recall_all", "recall_all"), ("ndcg", "ndcg")]:
        vals = []
        for k in [1, 3, 5, 10]:
            arr = metrics[k][key]
            vals.append(sum(arr) / len(arr) if arr else 0.0)
        print(f"  {row_name:<20} {vals[0]:>7.1%} {vals[1]:>7.1%} {vals[2]:>7.1%} {vals[3]:>7.1%}")

    avg_time = total_time / n if n else 0
    print(f"\n  Avg retrieval time: {avg_time*1000:.0f}ms")
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
