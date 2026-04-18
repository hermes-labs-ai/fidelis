"""
Full-pipeline cogito benchmark on LongMemEval.

Simulates the production /recall endpoint WITHOUT a running server:
  Stage 1 — recall_b: sub-query decomposition + RRF + cosine rerank (same as
             longmemeval_retrieval.py baseline), produces top-20 candidates.
  Stage 2 — LLM filter: cheap local model receives numbered candidates, outputs
             JSON integer indices only (integer-pointer pattern from recall.py).
             Model: qwen3.5:2b via native Ollama /api/chat (think:false).
  Scoring  — R@1, R@5, R@10, nDCG@1/5/10, plus per-stage latency.

Usage:
  python bench/longmemeval_full_pipeline.py --split s
  python bench/longmemeval_full_pipeline.py --split s --limit 50 --filter-model gemma3:4b
  python bench/longmemeval_full_pipeline.py --split s --top-k 20 --no-filter

Results saved to: bench/results-full-pipeline-2026-04-15.json
"""

import argparse
import json
import math
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (mirrored from recall_b.py / longmemeval_retrieval.py)
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

# Nomic task prefixes (trained into the model)
QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "

# Filter defaults
DEFAULT_FILTER_MODEL = "qwen3.5:2b"
FILTER_TIMEOUT = 30  # seconds — generous for small local model


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def batch_embed(texts: list[str], retries: int = 8) -> list[list[float]] | None:
    """Batch-embed via Ollama /api/embed. Returns list of vectors or None."""
    # 2000 chars = safe nomic context limit (~2 chars/token worst case)
    sanitized = [
        t[:2000] if len(t) > 2000 else (t if t.strip() else "empty")
        for t in texts
    ]
    for attempt in range(retries):
        try:
            body = json.dumps({
                "model": EMBED_MODEL,
                "input": sanitized,
                "options": {"num_ctx": 8192},
            }).encode()
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/embed",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            vecs = data.get("embeddings", [])
            return vecs if len(vecs) == len(texts) else None
        except Exception as e:
            if attempt < retries - 1:
                wait = min(2 ** attempt, 8)
                print(f"  [embed retry {attempt+1}/{retries}] {e} — waiting {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  [embed FAILED] {e}", file=sys.stderr)
                return None


def batch_embed_queries(texts: list[str]) -> list[list[float]] | None:
    """Embed query strings with search_query: prefix."""
    return batch_embed([QUERY_PREFIX + t for t in texts])


def batch_embed_docs(texts: list[str]) -> list[list[float]] | None:
    """Embed document strings with search_document: prefix."""
    return batch_embed([DOC_PREFIX + t for t in texts])


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Stage 1 — recall_b (sub-query decomposition + RRF + cosine rerank)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s-]", " ", text.lower()).split()


def key_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOP and len(t) >= 2]


def build_subqueries(query: str) -> list[str]:
    seen, result = set(), []

    def add(q: str):
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            result.append(q)

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


def stage1_retrieve(
    query: str,
    corpus_texts: list[str],
    corpus_vecs: list[list[float]],
    top_k: int = 20,
) -> tuple[list[int], list[float]]:
    """
    Run recall_b-style retrieval. Returns (ranked_indices, blended_scores).
    All indices are into corpus_texts; best first, up to top_k.
    Uses nomic query prefix for sub-query embedding.
    """
    subqueries = build_subqueries(query)

    # Embed all sub-queries in one batch (with query prefix)
    sq_vecs = batch_embed_queries(subqueries)
    if sq_vecs is None:
        # Fallback: try embedding just the original query
        sq_vecs = batch_embed_queries([query])
        if sq_vecs is None:
            # Total failure — return original order truncated to top_k
            return list(range(min(top_k, len(corpus_texts)))), []

    # Run each sub-query against corpus, get top-20 per sub-query
    runs: list[list[tuple[int, float]]] = []
    for sv in sq_vecs:
        scored = [(i, cosine_sim(sv, corpus_vecs[i])) for i in range(len(corpus_vecs))]
        scored.sort(key=lambda x: x[1], reverse=True)
        runs.append(scored[:20])

    # RRF merge
    rrf_scores: dict[int, float] = {}
    for run in runs:
        for rank, (idx, _) in enumerate(run, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank)

    # Cosine rerank: score each candidate against the original query (sq_vecs[0])
    query_vec = sq_vecs[0]
    cosine_scores = {i: cosine_sim(query_vec, corpus_vecs[i]) for i in rrf_scores}

    # Blend RRF + cosine
    rrf_max = max(rrf_scores.values()) if rrf_scores else 1.0
    blended: list[tuple[int, float]] = []
    for idx in rrf_scores:
        rrf_n = rrf_scores[idx] / rrf_max if rrf_max > 0 else 0.0
        cos = cosine_scores.get(idx, 0.0)
        score = (1.0 - _COSINE_WEIGHT) * rrf_n + _COSINE_WEIGHT * cos
        blended.append((idx, score))

    blended.sort(key=lambda x: x[1], reverse=True)
    top = blended[:top_k]
    return [idx for idx, _ in top], [s for _, s in top]


# ---------------------------------------------------------------------------
# Stage 2 — LLM filter (integer-pointer pattern from recall.py)
# ---------------------------------------------------------------------------

_FILTER_SYSTEM = (
    "You are a relevance filter for a memory retrieval system. "
    "Decide which numbered memories are relevant to the query. "
    "Output ONLY a JSON array of integers, ordered from most to least relevant. "
    "If the query is off-topic or none of the candidates are relevant, output []. "
    "No explanation, no other text. "
    "Examples: [1, 4, 7]   or   []"
)


def _parse_filter_indices(raw: str, n_candidates: int) -> list[int] | None:
    """
    Extract a JSON integer array from the model's output.
    Returns 0-based indices into the candidates list, or None on parse failure.
    Strips <think>...</think> blocks that qwen3 models emit.
    """
    # Strip thinking tokens
    if "<think>" in raw:
        end_think = raw.rfind("</think>")
        raw = raw[end_think + 8:].strip() if end_think >= 0 else raw

    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start < 0 or end <= start:
        return None

    try:
        indices = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None

    if not isinstance(indices, list):
        return None

    seen: set[int] = set()
    result: list[int] = []
    for idx in indices:
        # LLM outputs 1-based indices
        if isinstance(idx, int) and 1 <= idx <= n_candidates and idx not in seen:
            seen.add(idx)
            result.append(idx - 1)  # convert to 0-based

    return result


def stage2_filter(
    query: str,
    candidates: list[dict],  # [{"text": str, "score": float, "corpus_idx": int}]
    model: str,
    timeout: float = FILTER_TIMEOUT,
) -> tuple[list[dict], str]:
    """
    Run the LLM integer-pointer filter on candidates.
    Returns (filtered_candidates, method_tag).

    Uses native Ollama /api/chat with think:false for qwen3/qwen3.5 models.
    Falls back to OpenAI-compat /v1/chat/completions for non-thinking models.
    """
    if not candidates:
        return [], "filter_empty_input"

    # Build numbered candidate block (150 chars each, same as recall.py)
    lines = [
        f"[{i+1}] {c['text'][:150].replace(chr(10), ' ')}"
        for i, c in enumerate(candidates)
    ]
    candidates_block = "\n".join(lines)

    user = (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{candidates_block}\n\n"
        "Return a JSON array of the relevant memory numbers."
    )

    is_thinking = model.startswith("qwen3") or model.startswith("qwen3.5")

    if is_thinking:
        return _filter_ollama_native(model, user, candidates, timeout)
    else:
        return _filter_openai_compat(model, user, candidates, timeout)


def _filter_ollama_native(
    model: str,
    user: str,
    candidates: list[dict],
    timeout: float,
) -> tuple[list[dict], str]:
    """Use native Ollama /api/chat with think:false (required for qwen3/qwen3.5)."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _FILTER_SYSTEM},
            {"role": "user", "content": user},
        ],
        "think": False,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        raw = result["message"]["content"].strip()
        indices = _parse_filter_indices(raw, len(candidates))
        if indices is None:
            return candidates, "fallback_parse_error"
        if not indices:
            # Model said nothing is relevant — return empty (valid answer)
            return [], "filter_none_relevant"
        selected = [candidates[i] for i in indices]
        return selected, "filter"
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", str(e))
        return candidates, f"fallback_unreachable:{reason}"
    except Exception as e:
        return candidates, f"fallback_error:{type(e).__name__}"


def _filter_openai_compat(
    model: str,
    user: str,
    candidates: list[dict],
    timeout: float,
) -> tuple[list[dict], str]:
    """Use OpenAI-compat /v1/chat/completions (non-thinking models)."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _FILTER_SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": 150,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer ollama",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        raw = result["choices"][0]["message"]["content"].strip()
        indices = _parse_filter_indices(raw, len(candidates))
        if indices is None:
            return candidates, "fallback_parse_error"
        if not indices:
            return [], "filter_none_relevant"
        selected = [candidates[i] for i in indices]
        return selected, "filter"
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", str(e))
        return candidates, f"fallback_unreachable:{reason}"
    except Exception as e:
        return candidates, f"fallback_error:{type(e).__name__}"


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def recall_at_k(ranked_indices: list[int], correct_indices: set[int], k: int) -> float:
    """1.0 if any correct index is in top-k results, else 0.0."""
    return float(any(i in correct_indices for i in ranked_indices[:k]))


def ndcg_at_k(ranked_indices: list[int], correct_indices: set[int], n_corpus: int, k: int) -> float:
    relevances = [1 if i in correct_indices else 0 for i in range(n_corpus)]

    def dcg(rels: list[float]) -> float:
        if not rels:
            return 0.0
        val = rels[0]
        for j, r in enumerate(rels[1:], 2):
            val += r / math.log2(j)
        return val

    ranked_rel = [relevances[i] for i in ranked_indices[:k]]
    ideal_rel = sorted(relevances, reverse=True)[:k]
    ideal = dcg(ideal_rel)
    actual = dcg(ranked_rel)
    return actual / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="cogito full-pipeline (recall_b + LLM filter) on LongMemEval"
    )
    parser.add_argument("--split", choices=["s", "m"], default="s",
                        help="s=small (~50 sessions), m=medium (~500 sessions)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N questions (0=all)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Candidates passed from stage 1 to LLM filter (default: 20)")
    parser.add_argument("--filter-model", default=DEFAULT_FILTER_MODEL,
                        help=f"Ollama model for LLM filter stage (default: {DEFAULT_FILTER_MODEL})")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip LLM filter stage (recall_b only, as baseline comparison)")
    parser.add_argument("--data_dir",
                        default=str(Path(__file__).resolve().parent.parent.parent / "LongMemEval" / "data"))
    parser.add_argument("--out",
                        default=str(Path(__file__).parent / "results-full-pipeline-2026-04-15.json"))
    args = parser.parse_args()

    split_file = f"longmemeval_{args.split}_cleaned.json"
    data_path = Path(args.data_dir) / split_file
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading {data_path.name}...")
    with open(data_path) as f:
        data = json.load(f)

    # Filter out abstention questions (no retrieval target)
    data = [e for e in data if "_abs" not in e["question_id"]]
    if args.limit > 0:
        data = data[: args.limit]
    print(f"Running on {len(data)} questions (non-abstention)")
    if args.no_filter:
        print("  Mode: recall_b only (--no-filter)")
    else:
        print(f"  Mode: full pipeline (recall_b top-{args.top_k} → LLM filter [{args.filter_model}])")

    # Metrics accumulators — track both stage-1 and final (post-filter) rankings
    ks = [1, 5, 10]
    metrics_stage1 = {k: {"recall": [], "ndcg": []} for k in ks}
    metrics_final  = {k: {"recall": [], "ndcg": []} for k in ks}

    latency_embed = []
    latency_stage1 = []
    latency_filter = []

    filter_stats = {"filter": 0, "fallback": 0, "none_relevant": 0}
    per_question_results = []

    for qi, entry in enumerate(data):
        qid = entry["question_id"]
        question = entry["question"]
        answer_sids = set(entry["answer_session_ids"])

        # Build corpus: one text per session (user turns only)
        corpus_texts: list[str] = []
        corpus_ids: list[str] = []
        correct_indices: list[int] = []

        for sid, session in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
            text = " ".join(t["content"] for t in session if t["role"] == "user")
            corpus_texts.append(text)
            corpus_ids.append(sid)
            if sid in answer_sids:
                correct_indices.append(len(corpus_texts) - 1)

        if not correct_indices:
            continue

        # Embed corpus in batches of 100 (with document prefix)
        t_embed_start = time.time()
        all_vecs: list[list[float]] = []
        embed_failed = False
        for i in range(0, len(corpus_texts), 100):
            chunk = corpus_texts[i: i + 100]
            vecs = batch_embed_docs(chunk)
            if vecs is None:
                print(f"  [{qi+1}] EMBED FAILED for corpus chunk, skipping", file=sys.stderr)
                embed_failed = True
                break
            all_vecs.extend(vecs)

        if embed_failed:
            continue

        t_embed = time.time() - t_embed_start
        latency_embed.append(t_embed)

        # ------------------------------------------------------------------
        # Stage 1: recall_b retrieval → top-K candidates
        # ------------------------------------------------------------------
        t_s1 = time.time()
        stage1_indices, stage1_scores = stage1_retrieve(
            question, corpus_texts, all_vecs, top_k=args.top_k
        )
        latency_stage1.append(time.time() - t_s1)

        # Score stage-1 results
        correct_set = set(correct_indices)
        for k in ks:
            metrics_stage1[k]["recall"].append(recall_at_k(stage1_indices, correct_set, k))
            metrics_stage1[k]["ndcg"].append(ndcg_at_k(stage1_indices, correct_set, len(corpus_texts), k))

        # ------------------------------------------------------------------
        # Stage 2: LLM filter
        # ------------------------------------------------------------------
        if args.no_filter:
            final_indices = stage1_indices
            filter_method = "no_filter"
        else:
            # Build candidate dicts for the filter (text + original corpus index)
            stage1_candidates = [
                {
                    "text": corpus_texts[idx],
                    "score": round(score, 4),
                    "corpus_idx": idx,
                }
                for idx, score in zip(stage1_indices, stage1_scores)
            ]

            t_f = time.time()
            filtered_candidates, filter_method = stage2_filter(
                question,
                stage1_candidates,
                model=args.filter_model,
                timeout=FILTER_TIMEOUT,
            )
            latency_filter.append(time.time() - t_f)

            if filter_method == "filter":
                filter_stats["filter"] += 1
                final_indices = [c["corpus_idx"] for c in filtered_candidates]
            elif filter_method == "filter_none_relevant":
                filter_stats["none_relevant"] += 1
                # Model said nothing relevant — treat as empty (no retrieval)
                final_indices = []
            else:
                filter_stats["fallback"] += 1
                # On fallback, use stage-1 order unchanged
                final_indices = stage1_indices

        # Score final results
        for k in ks:
            metrics_final[k]["recall"].append(recall_at_k(final_indices, correct_set, k))
            metrics_final[k]["ndcg"].append(ndcg_at_k(final_indices, correct_set, len(corpus_texts), k))

        # Per-question record
        per_question_results.append({
            "qid": qid,
            "question": question,
            "correct_indices": correct_indices,
            "stage1_top10": stage1_indices[:10],
            "final_top10": final_indices[:10],
            "filter_method": filter_method,
            "stage1_r1": recall_at_k(stage1_indices, correct_set, 1),
            "final_r1": recall_at_k(final_indices, correct_set, 1),
            "stage1_r5": recall_at_k(stage1_indices, correct_set, 5),
            "final_r5": recall_at_k(final_indices, correct_set, 5),
        })

        # Progress output every 10 questions or on first
        if (qi + 1) % 10 == 0 or qi == 0:
            n_done = len(metrics_final[5]["recall"])
            run_r1 = sum(metrics_final[1]["recall"]) / n_done if n_done else 0
            run_r5 = sum(metrics_final[5]["recall"]) / n_done if n_done else 0
            last_r5 = metrics_final[5]["recall"][-1]
            status = "OK" if last_r5 > 0 else "--"
            avg_f = (sum(latency_filter) / len(latency_filter) * 1000) if latency_filter else 0
            print(
                f"  [{qi+1:4d}/{len(data)}] "
                f"R@1={run_r1:.1%}  R@5={run_r5:.1%}  "
                f"last={status}  filter={avg_f:.0f}ms avg  "
                f"method={filter_method}"
            )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    n = len(metrics_final[5]["recall"])
    if n == 0:
        print("No results scored.")
        sys.exit(1)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print(f"\n{'=' * 72}")
    label = f"cogito FULL PIPELINE on LongMemEval_{args.split.upper()}"
    if args.no_filter:
        label = f"cogito recall_b (no filter) on LongMemEval_{args.split.upper()}"
    print(f"  {label}  —  {n} questions")
    print(f"{'=' * 72}\n")

    # Metric table
    print(f"  {'Metric':<22} {'Stage1 (recall_b)':>20} {'Final (post-filter)':>22}")
    print(f"  {'─'*22} {'─'*20} {'─'*22}")
    for k in ks:
        r1_s1 = avg(metrics_stage1[k]["recall"])
        r1_fi = avg(metrics_final[k]["recall"])
        delta = r1_fi - r1_s1
        sign = "+" if delta >= 0 else ""
        print(f"  {'R@' + str(k):<22} {r1_s1:>19.1%}  {r1_fi:>19.1%}  ({sign}{delta:.1%})")

    print()
    for k in ks:
        nd_s1 = avg(metrics_stage1[k]["ndcg"])
        nd_fi = avg(metrics_final[k]["ndcg"])
        delta = nd_fi - nd_s1
        sign = "+" if delta >= 0 else ""
        print(f"  {'nDCG@' + str(k):<22} {nd_s1:>19.1%}  {nd_fi:>19.1%}  ({sign}{delta:.1%})")

    print()
    avg_embed  = avg(latency_embed) * 1000
    avg_s1     = avg(latency_stage1) * 1000
    avg_filt   = avg(latency_filter) * 1000 if latency_filter else 0
    total_avg  = avg_embed + avg_s1 + avg_filt
    print(f"  {'Latency (avg per question)':<22}")
    print(f"    embed (corpus):    {avg_embed:7.0f} ms")
    print(f"    stage1 (recall_b): {avg_s1:7.0f} ms")
    if not args.no_filter:
        print(f"    filter (LLM):      {avg_filt:7.0f} ms")
    print(f"    TOTAL pipeline:    {total_avg:7.0f} ms")

    if not args.no_filter:
        print(f"\n  Filter outcomes ({n} questions):")
        print(f"    filter (clean):    {filter_stats['filter']:4d}  ({filter_stats['filter']/n:.1%})")
        print(f"    none_relevant:     {filter_stats['none_relevant']:4d}  ({filter_stats['none_relevant']/n:.1%})")
        print(f"    fallback:          {filter_stats['fallback']:4d}  ({filter_stats['fallback']/n:.1%})")

    total_embed  = sum(latency_embed)
    total_filter = sum(latency_filter)
    est_total    = total_embed + sum(latency_stage1) + total_filter
    print(f"\n  Total runtime: {est_total/60:.1f} min")
    print(f"{'=' * 72}\n")

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_data = {
        "run_date": "2026-04-15",
        "split": args.split,
        "n_questions": n,
        "filter_model": args.filter_model if not args.no_filter else None,
        "top_k_candidates": args.top_k,
        "no_filter": args.no_filter,
        "metrics": {
            "stage1": {
                f"R@{k}": avg(metrics_stage1[k]["recall"]) for k in ks
            } | {
                f"nDCG@{k}": avg(metrics_stage1[k]["ndcg"]) for k in ks
            },
            "final": {
                f"R@{k}": avg(metrics_final[k]["recall"]) for k in ks
            } | {
                f"nDCG@{k}": avg(metrics_final[k]["ndcg"]) for k in ks
            },
        },
        "latency_ms": {
            "embed_avg": round(avg_embed, 1),
            "stage1_avg": round(avg_s1, 1),
            "filter_avg": round(avg_filt, 1),
            "total_avg": round(total_avg, 1),
        },
        "filter_stats": filter_stats if not args.no_filter else None,
        "per_question": per_question_results,
    }

    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
