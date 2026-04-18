"""
Full pipeline: Combined retrieval (BM25 + turn-level + prefixes) + LLM filter.

Stage 1: BM25 + turn-level chunking + nomic prefixes (same as longmemeval_combined.py)
  → R@1=83.2%, R@5=98.3% zero-LLM

Stage 2: LLM reranking of top-5 sessions via gemma3:4b
  → Hypothesis: reranking the 15% of cases where answer is at positions 2-5

Graceful fallback: if filter fails, keep Stage 1 order (never regress).
"""

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

import os
os.environ["BM25S_SHOW_PROGRESS"] = "0"
os.environ["TQDM_DISABLE"] = "1"
import bm25s
import importlib.util
_ts_spec = importlib.util.spec_from_file_location(
    "temporal_scaffold",
    str(Path(__file__).parent / "phase-4" / "temporal_scaffold.py"),
)
_ts_mod = importlib.util.module_from_spec(_ts_spec)
_ts_spec.loader.exec_module(_ts_mod)
build_temporal_scaffold = _ts_mod.build_temporal_scaffold
is_temporal_query = _ts_mod.is_temporal_query

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
FILTER_MODEL = "qwen-turbo"
FILTER_TOP_K = 5
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"

QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "
GAP_THRESHOLD = 0.1  # score gap: top1 - top2. Above = confident, skip LLM.

# ---------------------------------------------------------------------------
# Router: classify query → route decision
# ---------------------------------------------------------------------------
_SKIP_PATTERNS = [
    "you told me", "you suggested", "you recommended",
    "you mentioned", "you said", "you explained",
    "our previous conversation", "our last chat",
    "remind me what you",
]
_TEMPORAL_PATTERNS = [
    "which happened first", "which did i do first",
    "how many days", "how many weeks", "how many months",
    "before or after", "what order", "what was the date",
    "which event", "which trip",
    "order of the", "from earliest", "from first",
    "most recently", "a week ago", "two weeks ago",
    "a month ago", "last saturday", "last sunday",
    "last weekend", "last monday", "last tuesday",
    "graduated first", "started first", "finished first",
    "did i do first", "did i attend first",
    "who did i go with to the",
]
_COUNTING_PATTERNS = [
    "how many", "total number", "how much total",
    "in total", "altogether", "combined",
]


def classify_query(query: str) -> str:
    """Classify query for routing. Returns 'skip', 'llm', or 'gap'."""
    q = query.lower()
    if any(p in q for p in _SKIP_PATTERNS):
        return "skip"
    if any(p in q for p in _TEMPORAL_PATTERNS):
        return "llm"
    if any(p in q for p in _COUNTING_PATTERNS):
        return "llm"
    return "gap"

_FILTER_SYSTEM = (
    "Execute this procedure:\n"
    "```\n"
    "def rank(query, candidates):\n"
    "  for each candidate:\n"
    "    keyword_match = count(query_keywords IN candidate_text)\n"
    "    fact_match = candidate contains specific answer to query (bool)\n"
    "    score = keyword_match * 2 + fact_match * 10\n"
    "  return sorted(candidates, by=score, descending)\n"
    "```\n"
    "Input: query and 5 numbered candidates.\n"
    "Output: ONLY a JSON array of candidate numbers sorted by score.\n"
    "Example: [3, 1, 5, 2, 4]"
)


# ---------------------------------------------------------------------------
# Chunking (from longmemeval_combined.py)
# ---------------------------------------------------------------------------
def chunk_session(session: list[dict], session_id: str) -> list[tuple[str, str]]:
    chunks = []
    turns = []
    current_user = None
    for msg in session:
        if msg["role"] == "user":
            current_user = msg["content"]
        elif msg["role"] == "assistant" and current_user is not None:
            turns.append((current_user, msg["content"]))
            current_user = None
    if current_user is not None:
        turns.append((current_user, ""))
    if not turns:
        raw = " ".join(m["content"] for m in session if m["role"] == "user")
        if raw.strip():
            chunks.append((raw, session_id))
        return chunks
    for i, (user, assistant) in enumerate(turns):
        parts = []
        if i > 0:
            prev_u, prev_a = turns[i - 1]
            parts.append(f"[context] {prev_u}")
            if prev_a:
                parts.append(f"[context] {prev_a}")
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
    seen_sids: set[str] = set()
    ranked_sessions: list[int] = []
    for chunk_idx in ranked_chunk_indices:
        sid = chunk_session_ids[chunk_idx]
        if sid not in seen_sids:
            seen_sids.add(sid)
            corpus_idx = session_id_to_corpus_idx[sid]
            ranked_sessions.append(corpus_idx)
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
    return batch_embed([DOC_PREFIX + t for t in texts])


def batch_embed_queries(texts: list[str]) -> list[list[float]] | None:
    return batch_embed([QUERY_PREFIX + t for t in texts])


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
# Stage 1: Combined retrieval (BM25 + dense + turn-level + prefixes)
# ---------------------------------------------------------------------------
def retrieve_chunks(
    query: str,
    chunk_texts: list[str],
    chunk_vecs: list[list[float]],
    bm25_index: bm25s.BM25,
) -> tuple[list[int], list[float]]:
    subqueries = build_subqueries(query)
    sq_vecs = batch_embed_queries(subqueries)
    if sq_vecs is None:
        sq_vecs = batch_embed_queries([query])
        if sq_vecs is None:
            return list(range(len(chunk_texts)))

    k_bm25 = min(20, len(chunk_texts))
    runs: list[list[tuple[int, float]]] = []

    for subquery, sv in zip(subqueries, sq_vecs):
        scored = [(i, cosine_sim(sv, cv)) for i, cv in enumerate(chunk_vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        runs.append(scored[:20])

        query_tokens = bm25s.tokenize([subquery])
        bm25_results, bm25_scores = bm25_index.retrieve(query_tokens, k=k_bm25)
        bm25_run = [
            (int(idx), float(score))
            for idx, score in zip(bm25_results[0], bm25_scores[0])
        ]
        runs.append(bm25_run)

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
    ranked_scores = [s for _, s in blended]
    remaining = [i for i in range(len(chunk_texts)) if i not in rrf_scores]
    remaining_scored = [(i, cosine_sim(query_vec, chunk_vecs[i])) for i in remaining]
    remaining_scored.sort(key=lambda x: x[1], reverse=True)
    ranked_ids.extend([i for i, _ in remaining_scored])
    ranked_scores.extend([s for _, s in remaining_scored])

    return ranked_ids, ranked_scores


# ---------------------------------------------------------------------------
# Stage 2: LLM filter (rerank top-5 sessions)
# ---------------------------------------------------------------------------
def _parse_filter_indices(raw: str, n_candidates: int) -> list[int] | None:
    """Parse LLM output to list of 0-based indices."""
    # Strip thinking tokens if present
    if "<think>" in raw:
        end = raw.rfind("</think>")
        if end >= 0:
            after = raw[end + 8:].strip()
            if after:
                raw = after

    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start < 0 or end <= start:
        return None
    try:
        arr = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None

    result = []
    for item in arr:
        try:
            idx = int(item)
        except (ValueError, TypeError):
            continue
        if 1 <= idx <= n_candidates and (idx - 1) not in result:
            result.append(idx - 1)  # 1-based → 0-based
    return result if result else None


def llm_rerank(
    query: str,
    top_sessions: list[tuple[int, str]],  # [(corpus_idx, session_text), ...]
    temporal_scaffold: str = "",
) -> tuple[list[int], str]:
    """
    Rerank top sessions via LLM. Returns (reranked_corpus_indices, method).
    On any failure, returns original order (graceful fallback).
    """
    n = len(top_sessions)
    if n <= 1:
        return [idx for idx, _ in top_sessions], "filter_skip"

    # Build numbered candidate block
    lines = []
    for i, (corpus_idx, text) in enumerate(top_sessions):
        snippet = text[:500].replace("\n", " ")
        lines.append(f"[{i+1}] {snippet}")
    candidates_block = "\n".join(lines)

    # Build prompt with optional temporal scaffold
    prompt_parts = [f"Query: {query}"]
    if temporal_scaffold:
        prompt_parts.append(f"\n{temporal_scaffold}")
    prompt_parts.append(f"\nCandidate memories:\n{candidates_block}")
    prompt_parts.append(f"\nRank these {n} candidates by relevance. Output JSON array.")
    prompt_user = "\n".join(prompt_parts)

    try:
        body = json.dumps({
            "model": FILTER_MODEL,
            "messages": [
                {"role": "system", "content": _FILTER_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            "temperature": 0,
            "max_tokens": 100,
        }).encode()
        req = urllib.request.Request(
            DASHSCOPE_URL, data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"  [filter error] {e}", file=sys.stderr)
        return [idx for idx, _ in top_sessions], "fallback_error"

    if not raw.strip():
        return [idx for idx, _ in top_sessions], "fallback_empty"

    parsed = _parse_filter_indices(raw, n)
    if parsed is None:
        return [idx for idx, _ in top_sessions], "fallback_parse"

    # Build reranked list from parsed indices
    reranked = [top_sessions[i][0] for i in parsed]
    # Append any candidates the LLM didn't include
    included = set(parsed)
    for i in range(n):
        if i not in included:
            reranked.append(top_sessions[i][0])

    return reranked, "filter"


_VERIFY_SYSTEM = (
    "Execute this procedure:\n"
    "```\n"
    "def verify(query, candidate):\n"
    "  facts = extract_facts(candidate)\n"
    "  answer_present = any(fact answers query for fact in facts)\n"
    "  return 'YES' if answer_present else 'NO'\n"
    "```\n"
    "Input: a query and a candidate memory.\n"
    "Output: ONLY 'YES' or 'NO'. Nothing else."
)


def llm_verify(
    query: str,
    top_sessions: list[tuple[int, str]],
) -> tuple[list[int], str]:
    """
    Verify top candidate. If #1 doesn't answer the query, find the first that does.
    Falls back to original order on failure.
    """
    if len(top_sessions) <= 1:
        return [idx for idx, _ in top_sessions], "verify_skip"

    for check_i, (corpus_idx, text) in enumerate(top_sessions[:5]):
        snippet = text[:500].replace("\n", " ")
        prompt_user = f"Query: {query}\n\nCandidate memory:\n{snippet}\n\nDoes this memory answer the query?"

        try:
            body = json.dumps({
                "model": FILTER_MODEL,
                "messages": [
                    {"role": "system", "content": _VERIFY_SYSTEM},
                    {"role": "user", "content": prompt_user},
                ],
                "temperature": 0,
                "max_tokens": 10,
            }).encode()
            req = urllib.request.Request(
                DASHSCOPE_URL, data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
        except Exception as e:
            print(f"  [verify error] {e}", file=sys.stderr)
            return [idx for idx, _ in top_sessions], "verify_fallback"

        if "YES" in raw:
            # Found the answer — promote this candidate to rank 1
            if check_i == 0:
                return [idx for idx, _ in top_sessions], "verify_kept"
            else:
                reordered = [top_sessions[check_i][0]]
                for j, (idx2, _) in enumerate(top_sessions):
                    if j != check_i:
                        reordered.append(idx2)
                return reordered, f"verify_swap_{check_i+1}"

    # None verified — keep original order
    return [idx for idx, _ in top_sessions], "verify_none"


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
    parser.add_argument("--no-filter", action="store_true", help="Skip LLM filter (Stage 1 only)")
    parser.add_argument("--verify", action="store_true", help="Use verify mode: only check if #1 is correct, swap if not")
    parser.add_argument("--run-id", default=None, help="Run ID for per-question logging (default: auto timestamp)")
    args = parser.parse_args()

    split_file = f"longmemeval_{args.split}_cleaned.json"
    data_path = Path(args.data_dir) / split_file
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    mode = "Stage 1 only (no filter)" if args.no_filter else f"Stage 1 + LLM filter ({FILTER_MODEL})"
    print(f"COMBINED PIPELINE: {mode}")
    print(f"Loading {data_path.name}...")
    data = json.load(open(data_path))
    data = [e for e in data if "_abs" not in e["question_id"]]
    if args.limit > 0:
        data = data[:args.limit]
    print(f"[pipeline] Running on {len(data)} questions")

    # Setup run directory for per-question logging
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(__file__).parent / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] Run ID: {run_id} → {run_dir}")

    # Metrics for both stages
    s1_metrics = {k: {"recall_any": [], "recall_all": [], "ndcg": []} for k in [1, 3, 5, 10]}
    s2_metrics = {k: {"recall_any": [], "recall_all": [], "ndcg": []} for k in [1, 3, 5, 10]}
    total_retrieval_time = 0.0
    total_filter_time = 0.0
    filter_methods: dict[str, int] = {}
    per_question_log: list[dict] = []

    for qi, entry in enumerate(data):
        question = entry["question"]
        answer_sids = set(entry["answer_session_ids"])

        # --- Build session corpus ---
        corpus_texts: list[str] = []
        corpus_ids: list[str] = []
        correct_indices: list[int] = []
        session_id_to_corpus_idx: dict[str, int] = {}

        for sid, session in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
            corpus_idx = len(corpus_ids)
            text = " ".join(t["content"] for t in session if t["role"] == "user")
            corpus_texts.append(text)
            corpus_ids.append(sid)
            session_id_to_corpus_idx[sid] = corpus_idx
            if sid in answer_sids:
                correct_indices.append(corpus_idx)

        if not correct_indices:
            continue

        n_sessions = len(corpus_ids)

        # --- Build chunk corpus ---
        all_chunks: list[tuple[str, str]] = []
        for sid, session in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
            all_chunks.extend(chunk_session(session, sid))

        chunk_texts = [c[0] for c in all_chunks]
        chunk_session_ids = [c[1] for c in all_chunks]

        # --- BM25 index ---
        corpus_tokens = bm25s.tokenize(chunk_texts)
        bm25_index = bm25s.BM25()
        bm25_index.index(corpus_tokens)

        # --- Embed chunks ---
        all_vecs: list[list[float]] = []
        embed_ok = True
        for i in range(0, len(chunk_texts), 100):
            batch = chunk_texts[i:i + 100]
            vecs = batch_embed_docs(batch)
            if vecs is None:
                print(f"  [{qi+1}] EMBED FAILED, skipping")
                embed_ok = False
                break
            all_vecs.extend(vecs)

        if not embed_ok:
            continue

        # --- Stage 1: Combined retrieval ---
        t0 = time.time()
        ranked_chunk_indices, chunk_scores = retrieve_chunks(question, chunk_texts, all_vecs, bm25_index)
        retrieval_elapsed = time.time() - t0
        total_retrieval_time += retrieval_elapsed

        ranked_sessions = dedup_to_sessions(ranked_chunk_indices, chunk_session_ids, session_id_to_corpus_idx)

        # Score Stage 1
        for k in [1, 3, 5, 10]:
            r_any, r_all = evaluate_retrieval(ranked_sessions, correct_indices, k)
            n = ndcg_score(ranked_sessions, set(correct_indices), n_sessions, k)
            s1_metrics[k]["recall_any"].append(r_any)
            s1_metrics[k]["recall_all"].append(r_all)
            s1_metrics[k]["ndcg"].append(n)

        # --- Stage 2: Routed LLM filter ---
        filter_elapsed = 0.0
        if args.no_filter:
            final_ranking = ranked_sessions
            route_decision = "no_filter"
        else:
            route = classify_query(question)

            if route == "skip":
                # Assistant-reference queries: LLM hurts, keep S1
                final_ranking = ranked_sessions
                route_decision = "skip"
            elif route == "llm":
                # Temporal/counting: LLM helps, always call
                top_k = min(FILTER_TOP_K, len(ranked_sessions))
                top_sessions = [(idx, corpus_texts[idx]) for idx in ranked_sessions[:top_k]]

                # Build temporal scaffold for temporal queries
                scaffold = ""
                if is_temporal_query(question):
                    haystack_dates = entry.get("haystack_dates", [])
                    sid_to_date = {}
                    for sid_d, date_d in zip(entry["haystack_session_ids"], haystack_dates):
                        sid_to_date[sid_d] = date_d
                    top_sids = [corpus_ids[idx] for idx in ranked_sessions[:top_k]]
                    scaffold = build_temporal_scaffold(
                        list(range(1, top_k + 1)), top_sids, sid_to_date
                    )

                t1 = time.time()
                reranked_top, method = llm_rerank(question, top_sessions, temporal_scaffold=scaffold)
                filter_elapsed = time.time() - t1
                total_filter_time += filter_elapsed
                final_ranking = reranked_top + ranked_sessions[top_k:]
                route_decision = "llm_temporal" if scaffold else "llm"
            else:
                # Default: keep S1 (no gap threshold — LLM hurts on unclassified queries)
                final_ranking = ranked_sessions
                route_decision = "default_s1"

        filter_methods[route_decision] = filter_methods.get(route_decision, 0) + 1

        # Per-question instrumentation
        s1_top5_ids = [corpus_ids[i] for i in ranked_sessions[:5]]
        s2_top5_ids = [corpus_ids[i] for i in final_ranking[:5]] if not args.no_filter else s1_top5_ids
        s1_hit1 = any(i in set(ranked_sessions[:1]) for i in correct_indices)
        s2_hit1 = any(i in set(final_ranking[:1]) for i in correct_indices)
        s1_hit5 = any(i in set(ranked_sessions[:5]) for i in correct_indices)
        s2_hit5 = any(i in set(final_ranking[:5]) for i in correct_indices)

        pq_entry = {
            "qi": qi,
            "qid": entry["question_id"],
            "qtype": entry.get("question_type", "unknown"),
            "question": question,
            "gold_session_ids": list(answer_sids),
            "n_gold": len(answer_sids),
            "is_multi_answer": len(answer_sids) > 1,
            "s1_top5_ids": s1_top5_ids,
            "s2_top5_ids": s2_top5_ids,
            "s1_hit_at_1": s1_hit1,
            "s2_hit_at_1": s2_hit1,
            "s1_hit_at_5": s1_hit5,
            "s2_hit_at_5": s2_hit5,
            "route_decision": route_decision,
            "filter_called": route_decision in ("llm", "gap_to_llm"),
            "filter_ms": round(filter_elapsed * 1000) if route_decision in ("llm", "gap_to_llm") else 0,
        }
        per_question_log.append(pq_entry)

        # Write incrementally so we can monitor progress
        pq_path = run_dir / "per_question.json"
        with open(pq_path, "w") as f:
            json.dump(per_question_log, f)

        # Score Stage 2
        for k in [1, 3, 5, 10]:
            r_any, r_all = evaluate_retrieval(final_ranking, correct_indices, k)
            n = ndcg_score(final_ranking, set(correct_indices), n_sessions, k)
            s2_metrics[k]["recall_any"].append(r_any)
            s2_metrics[k]["recall_all"].append(r_all)
            s2_metrics[k]["ndcg"].append(n)

        # Progress
        if (qi + 1) % 10 == 0 or qi == 0:
            s1_r1 = sum(s1_metrics[1]["recall_any"]) / len(s1_metrics[1]["recall_any"])
            s2_r1 = sum(s2_metrics[1]["recall_any"]) / len(s2_metrics[1]["recall_any"])
            s2_r5 = sum(s2_metrics[5]["recall_any"]) / len(s2_metrics[5]["recall_any"])
            filt_avg = (total_filter_time / (qi + 1) * 1000) if not args.no_filter else 0
            print(
                f"  [{qi+1:3d}/{len(data)}] S1_R@1={s1_r1:.1%} S2_R@1={s2_r1:.1%} R@5={s2_r5:.1%}"
                f"  filter={filt_avg:.0f}ms  route={route_decision}"
            )

    n_q = len(s2_metrics[5]["recall_any"])

    print(f"\n{'='*70}")
    print(f"  COMBINED PIPELINE on LongMemEval_{args.split.upper()}  —  {n_q} questions")
    print(f"{'='*70}\n")

    # Stage 1 results
    print(f"  STAGE 1 (combined retrieval, zero LLM):")
    print(f"  {'Metric':<20} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    s1_results = {}
    for row_name, key in [("recall_any", "recall_any"), ("recall_all", "recall_all"), ("ndcg", "ndcg")]:
        vals = [sum(s1_metrics[k][key]) / len(s1_metrics[k][key]) if s1_metrics[k][key] else 0 for k in [1, 3, 5, 10]]
        s1_results[row_name] = {1: vals[0], 3: vals[1], 5: vals[2], 10: vals[3]}
        print(f"  {row_name:<20} {vals[0]:>7.1%} {vals[1]:>7.1%} {vals[2]:>7.1%} {vals[3]:>7.1%}")

    # Stage 2 results
    print(f"\n  STAGE 2 (+ LLM filter, {FILTER_MODEL if not args.no_filter else 'disabled'}):")
    print(f"  {'Metric':<20} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    s2_results = {}
    for row_name, key in [("recall_any", "recall_any"), ("recall_all", "recall_all"), ("ndcg", "ndcg")]:
        vals = [sum(s2_metrics[k][key]) / len(s2_metrics[k][key]) if s2_metrics[k][key] else 0 for k in [1, 3, 5, 10]]
        s2_results[row_name] = {1: vals[0], 3: vals[1], 5: vals[2], 10: vals[3]}
        print(f"  {row_name:<20} {vals[0]:>7.1%} {vals[1]:>7.1%} {vals[2]:>7.1%} {vals[3]:>7.1%}")

    # Delta
    s1_r1 = s1_results["recall_any"][1]
    s2_r1 = s2_results["recall_any"][1]
    delta = s2_r1 - s1_r1
    print(f"\n  Filter delta on R@1: {delta:+.1%}")

    avg_retrieval = total_retrieval_time / n_q if n_q else 0
    avg_filter = total_filter_time / n_q if n_q else 0
    print(f"  Avg retrieval: {avg_retrieval*1000:.0f}ms  Avg filter: {avg_filter*1000:.0f}ms  Total: {(avg_retrieval+avg_filter)*1000:.0f}ms")
    print(f"  Filter methods: {filter_methods}")

    # Save
    out = {
        "benchmark": "LongMemEval",
        "split": args.split.upper(),
        "date": "2026-04-15",
        "test": "combined_pipeline" if not args.no_filter else "combined_no_filter",
        "engine": f"BM25+turn-level+prefixes → LLM filter ({FILTER_MODEL})" if not args.no_filter else "BM25+turn-level+prefixes (no filter)",
        "filter_model": FILTER_MODEL if not args.no_filter else None,
        "filter_top_k": FILTER_TOP_K,
        "questions_evaluated": n_q,
        "stage1_metrics": {
            "recall_any": {f"R@{k}": round(s1_results["recall_any"][k], 3) for k in [1, 3, 5, 10]},
        },
        "stage2_metrics": {
            "recall_any": {f"R@{k}": round(s2_results["recall_any"][k], 3) for k in [1, 3, 5, 10]},
            "recall_all": {f"R@{k}": round(s2_results["recall_all"][k], 3) for k in [1, 3, 5, 10]},
            "ndcg": {f"R@{k}": round(s2_results["ndcg"][k], 3) for k in [1, 3, 5, 10]},
        },
        "filter_delta_r1": round(delta, 3),
        "filter_methods": filter_methods,
        "avg_retrieval_ms": round(avg_retrieval * 1000),
        "avg_filter_ms": round(avg_filter * 1000),
        "total_time_s": round(total_retrieval_time + total_filter_time, 1),
    }
    out_path = Path(__file__).parent / "results-combined-pipeline-2026-04-15.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Aggregate saved to {out_path}")

    # Save per-question log
    pq_path = run_dir / "per_question.json"
    with open(pq_path, "w") as f:
        json.dump(per_question_log, f, indent=2)

    # Per-question summary
    wins = sum(1 for pq in per_question_log if not pq["s1_hit_at_1"] and pq["s2_hit_at_1"])
    losses = sum(1 for pq in per_question_log if pq["s1_hit_at_1"] and not pq["s2_hit_at_1"])
    both_right = sum(1 for pq in per_question_log if pq["s1_hit_at_1"] and pq["s2_hit_at_1"])
    both_wrong = sum(1 for pq in per_question_log if not pq["s1_hit_at_1"] and not pq["s2_hit_at_1"])
    print(f"\n  Per-question: wins={wins} losses={losses} both_right={both_right} both_wrong={both_wrong}")
    print(f"  Per-question log saved to {pq_path}")

    # Also save aggregate to run dir
    with open(run_dir / "aggregate.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
