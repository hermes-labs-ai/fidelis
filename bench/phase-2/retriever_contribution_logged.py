"""
Step 5 runner: Modified pipeline that logs per-question retriever contributions.
For each question, logs: bm25_rank_of_gold, dense_rank_of_gold, rrf_rank_of_gold.

This is a SEPARATE script that runs the combined retrieval (no LLM filter)
with extra logging. Based on longmemeval_combined.py.

Outputs: bench/phase-2/retriever_contribution_detailed.json
"""

import json
import math
import re
import sys
import time
import os
import urllib.request
from pathlib import Path
from collections import defaultdict

os.environ["TQDM_DISABLE"] = "1"
import bm25s

# Same constants as combined pipeline
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
QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "


def batch_embed(texts, retries=8):
    sanitized = [t[:2000] if len(t) > 2000 else t if t.strip() else "empty" for t in texts]
    for attempt in range(retries):
        try:
            body = json.dumps({"model": EMBED_MODEL, "input": sanitized}).encode()
            req = urllib.request.Request(f"{OLLAMA_URL}/api/embed", data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            vecs = data.get("embeddings", [])
            return vecs if len(vecs) == len(texts) else None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 8))
            else:
                return None

def batch_embed_docs(texts): return batch_embed([DOC_PREFIX + t for t in texts])
def batch_embed_queries(texts): return batch_embed([QUERY_PREFIX + t for t in texts])

def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot/(na*nb) if na and nb else 0.0

def tokenize(text): return re.sub(r"[^\w\s-]", " ", text.lower()).split()
def key_tokens(text): return [t for t in tokenize(text) if t not in _STOP and len(t) >= 2]

def build_subqueries(query):
    seen, result = set(), []
    def add(q):
        q = q.strip()
        if q and q not in seen: seen.add(q); result.append(q)
    add(query)
    tokens = key_tokens(query)
    if not tokens: return result
    add(" ".join(tokens))
    for i in range(len(tokens)-1): add(f"{tokens[i]} {tokens[i+1]}")
    for i in range(len(tokens)-2): add(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
    for t in tokens:
        if len(t) >= 3: add(t)
    return result[:MAX_SUBQUERIES]

def chunk_session(session, session_id):
    chunks = []; turns = []; current_user = None
    for msg in session:
        if msg["role"] == "user": current_user = msg["content"]
        elif msg["role"] == "assistant" and current_user is not None:
            turns.append((current_user, msg["content"])); current_user = None
    if current_user: turns.append((current_user, ""))
    if not turns:
        raw = " ".join(m["content"] for m in session if m["role"] == "user")
        if raw.strip(): chunks.append((raw, session_id))
        return chunks
    for i, (user, asst) in enumerate(turns):
        parts = []
        if i > 0:
            parts.append(f"[context] {turns[i-1][0]}")
            if turns[i-1][1]: parts.append(f"[context] {turns[i-1][1]}")
        parts.append(user)
        if asst: parts.append(asst)
        chunks.append((" ".join(parts), session_id))
    return chunks


def retrieve_with_logging(query, chunk_texts, chunk_vecs, bm25_index, gold_chunk_indices):
    """
    Same as retrieve_chunks but returns per-retriever rank of gold.
    gold_chunk_indices: set of chunk indices that belong to gold sessions.
    """
    subqueries = build_subqueries(query)
    sq_vecs = batch_embed_queries(subqueries)
    if sq_vecs is None:
        return list(range(len(chunk_texts))), {"bm25": -1, "dense": -1, "rrf": -1}

    k_bm25 = min(20, len(chunk_texts))

    # Track best rank of any gold chunk per retriever
    best_dense_rank = 9999
    best_bm25_rank = 9999

    runs = []
    for subquery, sv in zip(subqueries, sq_vecs):
        # Dense
        scored = [(i, cosine_sim(sv, cv)) for i, cv in enumerate(chunk_vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        dense_run = scored[:20]
        runs.append(dense_run)
        for rank, (idx, _) in enumerate(scored):
            if idx in gold_chunk_indices:
                best_dense_rank = min(best_dense_rank, rank + 1)
                break

        # BM25
        query_tokens = bm25s.tokenize([subquery])
        bm25_results, bm25_scores = bm25_index.retrieve(query_tokens, k=k_bm25)
        bm25_run = [(int(idx), float(score)) for idx, score in zip(bm25_results[0], bm25_scores[0])]
        runs.append(bm25_run)
        for rank, (idx, _) in enumerate(bm25_run):
            if idx in gold_chunk_indices:
                best_bm25_rank = min(best_bm25_rank, rank + 1)
                break

    # RRF merge
    rrf_scores = {}
    for run in runs:
        for rank, (idx, _) in enumerate(run, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank)

    query_vec = sq_vecs[0]
    cosine_scores = {i: cosine_sim(query_vec, chunk_vecs[i]) for i in rrf_scores}
    rrf_max = max(rrf_scores.values()) if rrf_scores else 1.0

    blended = []
    for idx in rrf_scores:
        rrf_n = rrf_scores[idx] / rrf_max if rrf_max > 0 else 0.0
        cos = cosine_scores.get(idx, 0.0)
        score = (1.0 - _COSINE_WEIGHT) * rrf_n + _COSINE_WEIGHT * cos
        blended.append((idx, score))
    blended.sort(key=lambda x: x[1], reverse=True)

    # RRF rank of gold
    best_rrf_rank = 9999
    for rank, (idx, _) in enumerate(blended):
        if idx in gold_chunk_indices:
            best_rrf_rank = rank + 1
            break

    ranked_ids = [idx for idx, _ in blended]
    remaining = [i for i in range(len(chunk_texts)) if i not in rrf_scores]
    remaining_scored = [(i, cosine_sim(query_vec, chunk_vecs[i])) for i in remaining]
    remaining_scored.sort(key=lambda x: x[1], reverse=True)
    ranked_ids.extend([i for i, _ in remaining_scored])

    retriever_ranks = {
        "bm25": best_bm25_rank if best_bm25_rank < 9999 else -1,
        "dense": best_dense_rank if best_dense_rank < 9999 else -1,
        "rrf": best_rrf_rank if best_rrf_rank < 9999 else -1,
    }

    return ranked_ids, retriever_ranks


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(Path(__file__).resolve().parent.parent.parent / "LongMemEval" / "data"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data_path = Path(args.data_dir) / "longmemeval_s_cleaned.json"
    data = json.load(open(data_path))
    data = [e for e in data if "_abs" not in e["question_id"]]
    if args.limit > 0: data = data[:args.limit]

    print(f"RETRIEVER CONTRIBUTION LOGGING — {len(data)} questions")

    per_question = []
    for qi, entry in enumerate(data):
        question = entry["question"]
        answer_sids = set(entry["answer_session_ids"])

        # Build chunks
        all_chunks = []
        session_id_to_corpus_idx = {}
        for idx, (sid, session) in enumerate(zip(entry["haystack_session_ids"], entry["haystack_sessions"])):
            session_id_to_corpus_idx[sid] = idx
            all_chunks.extend(chunk_session(session, sid))

        chunk_texts = [c[0] for c in all_chunks]
        chunk_session_ids = [c[1] for c in all_chunks]

        # Gold chunk indices = chunks belonging to gold sessions
        gold_chunk_indices = {i for i, sid in enumerate(chunk_session_ids) if sid in answer_sids}

        # BM25 index
        corpus_tokens = bm25s.tokenize(chunk_texts)
        bm25_index = bm25s.BM25()
        bm25_index.index(corpus_tokens)

        # Embed
        all_vecs = []
        ok = True
        for i in range(0, len(chunk_texts), 100):
            vecs = batch_embed_docs(chunk_texts[i:i+100])
            if vecs is None: ok = False; break
            all_vecs.extend(vecs)
        if not ok: continue

        # Retrieve with logging
        _, ranks = retrieve_with_logging(question, chunk_texts, all_vecs, bm25_index, gold_chunk_indices)

        per_question.append({
            "qi": qi,
            "qid": entry["question_id"],
            "qtype": entry.get("question_type", "unknown"),
            "bm25_rank": ranks["bm25"],
            "dense_rank": ranks["dense"],
            "rrf_rank": ranks["rrf"],
        })

        if (qi + 1) % 50 == 0:
            print(f"  [{qi+1}/{len(data)}]")

    # Save
    out_path = Path(__file__).parent / "retriever_contribution_detailed.json"
    with open(out_path, "w") as f:
        json.dump(per_question, f, indent=2)

    # Aggregate
    print(f"\nRETRIEVER CONTRIBUTION SUMMARY ({len(per_question)} questions)")
    by_type = defaultdict(lambda: {"n": 0, "bm25_top1": 0, "bm25_top5": 0, "dense_top1": 0, "dense_top5": 0, "rrf_top1": 0, "rrf_top5": 0})
    for q in per_question:
        t = q["qtype"]
        by_type[t]["n"] += 1
        if q["bm25_rank"] == 1: by_type[t]["bm25_top1"] += 1
        if 1 <= q["bm25_rank"] <= 5: by_type[t]["bm25_top5"] += 1
        if q["dense_rank"] == 1: by_type[t]["dense_top1"] += 1
        if 1 <= q["dense_rank"] <= 5: by_type[t]["dense_top5"] += 1
        if q["rrf_rank"] == 1: by_type[t]["rrf_top1"] += 1
        if 1 <= q["rrf_rank"] <= 5: by_type[t]["rrf_top5"] += 1

    print(f"{'Type':35s} {'n':>4s} {'BM25@1':>7s} {'BM25@5':>7s} {'Dense@1':>8s} {'Dense@5':>8s} {'RRF@1':>6s} {'RRF@5':>6s}")
    for t, v in sorted(by_type.items(), key=lambda x: -x[1]["n"]):
        n = v["n"]
        print(f"{t:35s} {n:4d} {v['bm25_top1']/n:7.0%} {v['bm25_top5']/n:7.0%} {v['dense_top1']/n:8.0%} {v['dense_top5']/n:8.0%} {v['rrf_top1']/n:6.0%} {v['rrf_top5']/n:6.0%}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
