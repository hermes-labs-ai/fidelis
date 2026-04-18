"""
Test 1: Multi-query expansion via LLM (gemma3:4b).

For each question, generate 3 query variants (rephrase, keywords, inverted),
then search all variants + original, merge results with RRF.

Based on longmemeval_retrieval.py baseline.
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
LLM_MODEL = "gemma3:4b"


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def batch_embed(texts: list[str], retries: int = 8) -> list[list[float]] | None:
    sanitized = [t[:4000] if len(t) > 4000 else t if t.strip() else "empty" for t in texts]
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
                wait = min(2 ** attempt, 30)
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
# LLM query expansion
# ---------------------------------------------------------------------------
def llm_expand_query(query: str) -> list[str]:
    """Generate 3 query variants via gemma3:4b."""
    prompt = (
        "Generate exactly 3 rephrasings of this question. "
        "Line 1: rephrase using different words. "
        "Line 2: extract key terms only (no sentence). "
        "Line 3: rephrase focusing on the answer type expected.\n"
        "Return ONLY 3 lines, no numbering, no explanation.\n\n"
        f"{query}"
    )
    try:
        body = json.dumps({
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 150},
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        content = data.get("message", {}).get("content", "")
        lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
        # Clean numbering if present
        cleaned = []
        for l in lines[:3]:
            l = re.sub(r"^\d+[\.\)]\s*", "", l)
            if l and len(l) > 5:
                cleaned.append(l)
        return cleaned
    except Exception as e:
        print(f"  [llm expand error] {e}", file=sys.stderr)
        return []


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
        if len(t) >= 4:
            add(t)
    return result[:MAX_SUBQUERIES]


# ---------------------------------------------------------------------------
# Retrieval: LLM multi-query expansion + sub-query decomposition + RRF
# ---------------------------------------------------------------------------
def retrieve_sessions(query: str, corpus_texts: list[str], corpus_vecs: list[list[float]]) -> list[int]:
    # Step 1: LLM-generated query variants
    llm_variants = llm_expand_query(query)

    # Step 2: Build all queries = original subqueries + LLM variant subqueries
    all_queries = build_subqueries(query)
    for variant in llm_variants:
        for sq in build_subqueries(variant)[:4]:  # limit per variant
            if sq not in set(all_queries):
                all_queries.append(sq)

    # Cap total queries
    all_queries = all_queries[:16]

    # Embed all queries in one batch
    sq_vecs = batch_embed(all_queries)
    if sq_vecs is None:
        sq_vecs = batch_embed([query])
        if sq_vecs is None:
            return list(range(len(corpus_texts)))

    # Run each query against corpus via cosine similarity
    runs: list[list[tuple[int, float]]] = []
    for sv in sq_vecs:
        scored = [(i, cosine_sim(sv, cv)) for i, cv in enumerate(corpus_vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        runs.append(scored[:20])

    # RRF merge
    rrf_scores: dict[int, float] = {}
    for run in runs:
        for rank, (idx, _score) in enumerate(run, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank)

    # Cosine rerank against original query
    query_vec = sq_vecs[0]
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

    ranked_ids = [idx for idx, _ in blended]
    remaining = [i for i in range(len(corpus_texts)) if i not in rrf_scores]
    remaining_scored = [(i, cosine_sim(query_vec, corpus_vecs[i])) for i in remaining]
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

    print(f"Loading {data_path.name}...")
    data = json.load(open(data_path))
    data = [e for e in data if "_abs" not in e["question_id"]]
    if args.limit > 0:
        data = data[:args.limit]
    print(f"[multiquery] Running on {len(data)} questions with LLM expansion via {LLM_MODEL}")

    # Warmup: force nomic model load
    for _ in range(10):
        if batch_embed(["warmup"]):
            print(f"[multiquery] nomic-embed-text ready")
            break
        time.sleep(3)

    metrics = {k: {"recall_any": [], "recall_all": [], "ndcg": []} for k in [1, 3, 5, 10]}
    total_time = 0.0
    llm_fail_count = 0

    for qi, entry in enumerate(data):
        question = entry["question"]
        answer_sids = set(entry["answer_session_ids"])

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

        all_vecs = []
        for i in range(0, len(corpus_texts), 100):
            chunk = corpus_texts[i:i+100]
            vecs = batch_embed(chunk)
            if vecs is None:
                print(f"  [{qi+1}] EMBED FAILED for corpus chunk, skipping")
                break
            all_vecs.extend(vecs)
        else:
            t0 = time.time()
            rankings = retrieve_sessions(question, corpus_texts, all_vecs)
            elapsed = time.time() - t0
            total_time += elapsed

            for k in [1, 3, 5, 10]:
                r_any, r_all = evaluate_retrieval(rankings, correct_indices, k)
                n = ndcg_score(rankings, set(correct_indices), len(corpus_texts), k)
                metrics[k]["recall_any"].append(r_any)
                metrics[k]["recall_all"].append(r_all)
                metrics[k]["ndcg"].append(n)

            r5 = metrics[5]["recall_any"][-1]
            status = "✅" if r5 > 0 else "❌"
            if (qi + 1) % 20 == 0 or qi == 0:
                running_r1 = sum(metrics[1]["recall_any"]) / len(metrics[1]["recall_any"])
                running_r5 = sum(metrics[5]["recall_any"]) / len(metrics[5]["recall_any"])
                print(f"  [{qi+1}/{len(data)}] R@1={running_r1:.1%} R@5={running_r5:.1%}  last={status}  {elapsed:.1f}s  q={question[:50]}")
            continue
        continue

    n = len(metrics[5]["recall_any"])
    print(f"\n{'='*70}")
    print(f"  MULTIQUERY: cogito recall_b + LLM expansion on LongMemEval_{args.split.upper()}  —  {n} questions")
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

    # Save JSON results
    out = {
        "benchmark": "LongMemEval",
        "split": args.split.upper(),
        "date": "2026-04-15",
        "test": "multiquery_expansion",
        "engine": "recall_b + LLM multi-query expansion (gemma3:4b)",
        "embed_model": "nomic-embed-text",
        "llm_model": LLM_MODEL,
        "questions_evaluated": n,
        "metrics": {
            "recall_any": {f"R@{k}": round(results["recall_any"][k], 3) for k in [1, 3, 5, 10]},
            "recall_all": {f"R@{k}": round(results["recall_all"][k], 3) for k in [1, 3, 5, 10]},
            "ndcg": {f"R@{k}": round(results["ndcg"][k], 3) for k in [1, 3, 5, 10]},
        },
        "avg_retrieval_time_ms": round(avg_time * 1000),
        "total_time_s": round(total_time, 1),
    }
    out_path = Path(__file__).parent / "results-multiquery-2026-04-15.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
