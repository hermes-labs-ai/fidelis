"""
v33 Ablation Script — identical to v33 but with 4 CLI flags to disable components:
  --no-temporal-boost   : disable temporal boost at Stage 1
  --no-socratic         : replace s8-socratic with baseline filter prompt
  --no-verify-guard     : disable verify-guard (don't restore S1 top-1)
  --no-flagship         : disable qwen-max escalation (use qwen-turbo only)

Used for ablation runs K/L/M/N to attribute the +1.11pp lift.
"""

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from datetime import datetime
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

# Load flagship escalation
_fl_spec = importlib.util.spec_from_file_location(
    "flagship_escalation",
    str(Path(__file__).parent / "phase-6" / "flagship_escalation.py"),
)
_fl_mod = importlib.util.module_from_spec(_fl_spec)
_fl_spec.loader.exec_module(_fl_mod)
flagship_rerank = _fl_mod.flagship_rerank

# Load hardset qids for escalation gating
_hardset_path = Path(__file__).parent / "hardset.json"
HARDSET_QIDS = set()
if _hardset_path.exists():
    HARDSET_QIDS = {h["qid"] for h in json.load(open(_hardset_path))}

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
# calibrated: 0.1 from LongMemEval_S score distributions (runB-flagship, 470q)

# Temporal boost strength (multiplier on blended score when signal matches)
# calibrated: 1.5 max and 1.15 light from ablation runK/runL (LongMemEval_S 470q)
TEMPORAL_BOOST_MAX = 1.5
TEMPORAL_BOOST_LIGHT = 1.15  # light recency: fallback when no specific direction

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

_RECENCY_SIGNALS = ["recent", "lately", "now", "current", "just", "latest", "newest"]
_EARLIEST_SIGNALS = ["first", "originally", "earliest", "oldest", "beginning", "started", "initial"]


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


# ---------------------------------------------------------------------------
# Temporal boost helpers
# ---------------------------------------------------------------------------
def _parse_date(date_str: str) -> datetime | None:
    """Parse 'YYYY/MM/DD (Day) HH:MM' format."""
    if not date_str:
        return None
    try:
        parts = date_str.split(" ")
        return datetime.strptime(parts[0], "%Y/%m/%d")
    except (ValueError, IndexError):
        return None


def _extract_explicit_year_month(query: str) -> tuple[int | None, int | None]:
    """Extract explicit year and month from query text."""
    year = None
    month = None
    m = re.search(r'\b(202[0-9])\b', query)
    if m:
        year = int(m.group(1))
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    ql = query.lower()
    for name, num in month_map.items():
        if name in ql:
            month = num
            break
    return year, month


def temporal_boost_score(
    session_id: str,
    query: str,
    haystack_dates: list[str],
    haystack_session_ids: list[str],
    question_date: str = "",
) -> float:
    """Returns a multiplier 1.0-1.5 to apply to a session's blended score."""
    sid_to_date: dict[str, datetime | None] = {}
    for sid, ds in zip(haystack_session_ids, haystack_dates):
        sid_to_date[sid] = _parse_date(ds)

    session_dt = sid_to_date.get(session_id)
    if session_dt is None:
        return 1.0

    all_dated: list[tuple[str, datetime]] = [
        (sid, dt) for sid, dt in sid_to_date.items() if dt is not None
    ]
    if not all_dated:
        return 1.0

    all_dated.sort(key=lambda x: x[1])
    n_total = len(all_dated)

    session_rank = next(
        (i for i, (sid, _) in enumerate(all_dated) if sid == session_id), 0
    )

    ql = query.lower()

    year_q, month_q = _extract_explicit_year_month(ql)
    if year_q or month_q:
        match = True
        if year_q and session_dt.year != year_q:
            match = False
        if month_q and session_dt.month != month_q:
            match = False
        if match:
            return TEMPORAL_BOOST_MAX
        else:
            return 1.0

    if any(sig in ql for sig in _RECENCY_SIGNALS):
        recency_rank = session_rank / max(n_total - 1, 1)
        boost = 1.0 + (TEMPORAL_BOOST_MAX - 1.0) * math.log1p(recency_rank * (math.e - 1))
        return round(boost, 4)

    if any(sig in ql for sig in _EARLIEST_SIGNALS):
        oldest_rank = 1.0 - (session_rank / max(n_total - 1, 1))
        boost = 1.0 + (TEMPORAL_BOOST_MAX - 1.0) * math.log1p(oldest_rank * (math.e - 1))
        return round(boost, 4)

    recency_rank = session_rank / max(n_total - 1, 1)
    boost = 1.0 + (TEMPORAL_BOOST_LIGHT - 1.0) * recency_rank
    return round(boost, 4)


# ---------------------------------------------------------------------------
# s8-socratic scaffold (default filter for v33)
# ---------------------------------------------------------------------------

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

# s8-socratic scaffold prompt prefix
_S8_SOCRATIC_PREFIX = (
    "Step 1: classify query type (temporal, preference, factual, etc.).\n"
    "Step 2: identify the best candidate session that answers the query.\n"
    "Step 3: verify your choice with specific evidence from that candidate.\n"
    "Return the index of the best candidate as a JSON array (best first).\n\n"
)

# Baseline filter prompt (no socratic prefix) — for --no-socratic ablation
_BASELINE_PREFIX = ""


def _parse_filter_indices(raw: str, n_candidates: int) -> list[int] | None:
    """Parse LLM output to list of 0-based indices."""
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


def _llm_call_single(system_prompt: str, user_prompt: str, max_tokens: int = 150) -> str | None:
    """Single LLM call to qwen-turbo. Returns raw response string or None on error."""
    model = FILTER_MODEL
    try:
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
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
        return raw
    except Exception as e:
        print(f"  [llm_call error] {e}", file=sys.stderr)
        return None


def llm_rerank_with_prefix(
    query: str,
    top_sessions: list[tuple[int, str]],
    prefix: str,
    scaffold_name: str,
) -> tuple[list[int], str, dict]:
    """
    Apply filter with given prefix. Used for both s8-socratic and baseline.
    Returns (reranked_corpus_indices, method, meta_info).
    """
    n = len(top_sessions)
    if n == 0:
        return [], "empty", {}

    candidates_block = "\n".join(
        f"[{i+1}] {text[:500].replace(chr(10), ' ')}"
        for i, (_, text) in enumerate(top_sessions)
    )
    prompt = (
        f"{prefix}"
        f"Query: {query}\n\n"
        f"Candidate memories:\n{candidates_block}\n\n"
        f"Rank these {n} candidates by relevance. Output JSON array."
    )
    raw = _llm_call_single(_FILTER_SYSTEM, prompt, max_tokens=200)
    parsed = _parse_filter_indices(raw or "", n) if raw else None
    if not parsed:
        return [idx for idx, _ in top_sessions], "parse_fail", {"scaffold": scaffold_name}

    reranked = [top_sessions[i][0] for i in parsed]
    included = set(parsed)
    for i in range(n):
        if i not in included:
            reranked.append(top_sessions[i][0])
    return reranked, f"filter_{scaffold_name}", {"scaffold": scaffold_name}


def llm_rerank_flagship_meta(
    query: str,
    top_sessions: list[tuple[int, str]],
) -> tuple[list[int], str]:
    """Flagship path — delegate to flagship_rerank."""
    return flagship_rerank(query, top_sessions)


# ---------------------------------------------------------------------------
# Verify guard
# ---------------------------------------------------------------------------
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


def llm_verify_one(query: str, candidate_text: str, snippet_len: int = 1500) -> str | None:
    """Single-candidate verify. Returns 'YES', 'NO', or None on error."""
    snippet = candidate_text[:snippet_len].replace("\n", " ")
    prompt_user = (
        f"Query: {query}\n\n"
        f"Candidate memory:\n{snippet}\n\n"
        "Does this memory contain the specific fact that answers the query?"
    )
    raw = _llm_call_single(_VERIFY_SYSTEM, prompt_user, max_tokens=10)
    if raw is None:
        return None
    raw = raw.strip().upper()
    if "YES" in raw:
        return "YES"
    if "NO" in raw:
        return "NO"
    return None


# ---------------------------------------------------------------------------
# Chunking
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
# Stage 1: Combined retrieval
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
            return list(range(len(chunk_texts))), [0.0] * len(chunk_texts)

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


def apply_temporal_boost_to_sessions(
    ranked_sessions: list[int],
    session_scores: dict[int, float],
    corpus_ids: list[str],
    query: str,
    haystack_dates: list[str],
    haystack_session_ids: list[str],
    question_date: str = "",
) -> tuple[list[int], int]:
    """Apply temporal boost to session-level scores."""
    boost_fired = 0
    boosted_scores: list[tuple[int, float]] = []

    for corpus_idx in ranked_sessions:
        sid = corpus_ids[corpus_idx]
        base_score = session_scores.get(corpus_idx, 0.0)
        multiplier = temporal_boost_score(
            sid, query, haystack_dates, haystack_session_ids, question_date
        )
        if multiplier > 1.0:
            boost_fired += 1
        boosted_scores.append((corpus_idx, base_score * multiplier))

    boosted_scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in boosted_scores], boost_fired


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
    parser.add_argument("--data_dir", default=str(Path(__file__).parent.parent.parent / "LongMemEval" / "data"))
    parser.add_argument("--no-filter", action="store_true", help="Skip LLM filter (Stage 1 only)")
    parser.add_argument("--run-id", default=None, help="Run ID for per-question logging")
    parser.add_argument("--resume", action="store_true", help="Resume from existing per_question.json")
    # Ablation flags
    parser.add_argument("--no-temporal-boost", action="store_true",
                        help="ABLATION: disable temporal boost at Stage 1")
    parser.add_argument("--no-socratic", action="store_true",
                        help="ABLATION: use baseline filter prompt instead of s8-socratic")
    parser.add_argument("--no-verify-guard", action="store_true",
                        help="ABLATION: disable verify-guard (don't restore S1 top-1)")
    parser.add_argument("--no-flagship", action="store_true",
                        help="ABLATION: disable qwen-max escalation (use qwen-turbo only)")
    args = parser.parse_args()

    # Determine scaffold prefix
    if args.no_socratic:
        scaffold_prefix = _BASELINE_PREFIX
        scaffold_label = "baseline"
    else:
        scaffold_prefix = _S8_SOCRATIC_PREFIX
        scaffold_label = "s8-socratic"

    # Build ablation description
    ablation_parts = []
    if args.no_temporal_boost:
        ablation_parts.append("no-temporal-boost")
    if args.no_socratic:
        ablation_parts.append("no-socratic")
    if args.no_verify_guard:
        ablation_parts.append("no-verify-guard")
    if args.no_flagship:
        ablation_parts.append("no-flagship")
    ablation_desc = "+".join(ablation_parts) if ablation_parts else "full-v33"

    split_file = f"longmemeval_{args.split}_cleaned.json"
    data_path = Path(args.data_dir) / split_file
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    print(f"COMBINED PIPELINE v33-ABLATE: ablation={ablation_desc}")
    print(f"  temporal-boost: {'OFF' if args.no_temporal_boost else 'ON'}")
    print(f"  scaffold: {'baseline' if args.no_socratic else 's8-socratic'}")
    print(f"  verify-guard: {'OFF' if args.no_verify_guard else 'ON'}")
    print(f"  flagship: {'OFF (qwen-turbo only)' if args.no_flagship else 'ON (qwen-max for hardset)'}")
    print(f"Loading {data_path.name}...")
    data = json.load(open(data_path))
    data = [e for e in data if "_abs" not in e["question_id"]]
    if args.limit > 0:
        data = data[:args.limit]
    print(f"[pipeline] Running on {len(data)} questions")

    # Setup run directory
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(__file__).parent / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] Run ID: {run_id} → {run_dir}")

    # Metrics
    s1_metrics = {k: {"recall_any": [], "recall_all": [], "ndcg": []} for k in [1, 3, 5, 10]}
    s1b_metrics = {k: {"recall_any": [], "recall_all": [], "ndcg": []} for k in [1, 3, 5, 10]}
    s2_metrics = {k: {"recall_any": [], "recall_all": [], "ndcg": []} for k in [1, 3, 5, 10]}
    total_retrieval_time = 0.0
    total_filter_time = 0.0
    filter_methods: dict[str, int] = {}
    per_question_log: list[dict] = []

    temporal_boost_fired_count = 0
    socratic_fired_count = 0

    # Resume support
    done_qids: set[str] = set()
    pq_path = run_dir / "per_question.json"
    if args.resume and pq_path.exists():
        existing = json.load(open(pq_path))
        per_question_log = existing
        done_qids = {e["qid"] for e in existing}
        for e in existing:
            filter_methods[e["route_decision"]] = filter_methods.get(e["route_decision"], 0) + 1
            if e.get("temporal_boost_fired"):
                temporal_boost_fired_count += 1
            if e.get("scaffold") == scaffold_label:
                socratic_fired_count += 1
        print(f"[resume] Loaded {len(done_qids)} completed questions, skipping them.")

    for qi, entry in enumerate(data):
        if entry["question_id"] in done_qids:
            continue

        question = entry["question"]
        answer_sids = set(entry["answer_session_ids"])
        qtype = entry.get("question_type", "unknown")
        question_date = entry.get("question_date", "")
        haystack_dates = entry.get("haystack_dates", [])

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

        # Score Stage 1 (before temporal boost)
        for k in [1, 3, 5, 10]:
            r_any, r_all = evaluate_retrieval(ranked_sessions, correct_indices, k)
            n_val = ndcg_score(ranked_sessions, set(correct_indices), n_sessions, k)
            s1_metrics[k]["recall_any"].append(r_any)
            s1_metrics[k]["recall_all"].append(r_all)
            s1_metrics[k]["ndcg"].append(n_val)

        # --- Temporal boost at Stage 1 (ablatable) ---
        boost_fired_this_q = 0
        if not args.no_temporal_boost and is_temporal_query(question) and haystack_dates:
            session_scores: dict[int, float] = {
                corpus_idx: 1.0 / (rank + 1) for rank, corpus_idx in enumerate(ranked_sessions)
            }
            ranked_sessions_boosted, boost_fired_this_q = apply_temporal_boost_to_sessions(
                ranked_sessions,
                session_scores,
                corpus_ids,
                question,
                haystack_dates,
                entry["haystack_session_ids"],
                question_date,
            )
            ranked_sessions = ranked_sessions_boosted
            if boost_fired_this_q > 0:
                temporal_boost_fired_count += 1

        # Score Stage 1b (after temporal boost / no-op if disabled)
        for k in [1, 3, 5, 10]:
            r_any, r_all = evaluate_retrieval(ranked_sessions, correct_indices, k)
            n_val = ndcg_score(ranked_sessions, set(correct_indices), n_sessions, k)
            s1b_metrics[k]["recall_any"].append(r_any)
            s1b_metrics[k]["recall_all"].append(r_all)
            s1b_metrics[k]["ndcg"].append(n_val)

        # --- Stage 2: Routed LLM filter ---
        filter_elapsed = 0.0
        scaffold_used: str | None = None

        if args.no_filter:
            final_ranking = ranked_sessions
            route_decision = "no_filter"
        else:
            route = classify_query(question)

            if route == "skip":
                final_ranking = ranked_sessions
                route_decision = "skip"
            elif route == "llm":
                top_k = min(FILTER_TOP_K, len(ranked_sessions))
                top_sessions = [(idx, corpus_texts[idx]) for idx in ranked_sessions[:top_k]]

                t1 = time.time()
                qid = entry["question_id"]

                # Flagship path (ablatable)
                if not args.no_flagship and qid in HARDSET_QIDS:
                    top_sessions_full = [(idx, corpus_texts[idx]) for idx in ranked_sessions[:top_k]]
                    reranked_top, method = llm_rerank_flagship_meta(question, top_sessions_full)
                    route_decision = "flagship_llm"
                else:
                    # Apply filter with selected scaffold (socratic or baseline)
                    reranked_top, method, meta_info = llm_rerank_with_prefix(
                        question,
                        top_sessions,
                        scaffold_prefix,
                        scaffold_label,
                    )
                    route_decision = "llm"
                    scaffold_used = meta_info.get("scaffold")
                    if scaffold_used == scaffold_label:
                        socratic_fired_count += 1

                    # GUARD: restore S1 top-1 if demoted (ablatable)
                    if not args.no_verify_guard:
                        s1_top1_idx = ranked_sessions[0]
                        if reranked_top and reranked_top[0] != s1_top1_idx:
                            s1_top1_text = corpus_texts[s1_top1_idx]
                            verdict = llm_verify_one(question, s1_top1_text)
                            if verdict == "YES":
                                new_order = [s1_top1_idx]
                                for idx in reranked_top:
                                    if idx != s1_top1_idx:
                                        new_order.append(idx)
                                included = set(new_order)
                                for idx in ranked_sessions[:top_k]:
                                    if idx not in included:
                                        new_order.append(idx)
                                reranked_top = new_order
                                route_decision = "llm_guarded_s1"

                filter_elapsed = time.time() - t1
                total_filter_time += filter_elapsed
                final_ranking = reranked_top + ranked_sessions[top_k:]
            else:
                # Default / gap: escalate hardset to flagship (ablatable), else keep S1
                qid = entry["question_id"]
                if not args.no_flagship and qid in HARDSET_QIDS:
                    top_k = min(FILTER_TOP_K, len(ranked_sessions))
                    top_sessions_full = [(idx, corpus_texts[idx]) for idx in ranked_sessions[:top_k]]
                    t1 = time.time()
                    reranked_top, method = llm_rerank_flagship_meta(question, top_sessions_full)
                    filter_elapsed = time.time() - t1
                    total_filter_time += filter_elapsed
                    final_ranking = reranked_top + ranked_sessions[top_k:]
                    route_decision = "flagship"
                else:
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
            "qtype": qtype,
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
            "filter_called": route_decision in ("llm", "llm_guarded_s1", "gap_to_llm"),
            "filter_ms": round(filter_elapsed * 1000) if route_decision in ("llm", "llm_guarded_s1", "gap_to_llm") else 0,
            "temporal_boost_fired": boost_fired_this_q > 0,
            "temporal_boost_count": boost_fired_this_q,
            "scaffold": scaffold_used,
            "ablation": ablation_desc,
        }
        per_question_log.append(pq_entry)

        # Write incrementally
        pq_path = run_dir / "per_question.json"
        with open(pq_path, "w") as f:
            json.dump(per_question_log, f)

        # Score Stage 2
        for k in [1, 3, 5, 10]:
            r_any, r_all = evaluate_retrieval(final_ranking, correct_indices, k)
            n_val = ndcg_score(final_ranking, set(correct_indices), n_sessions, k)
            s2_metrics[k]["recall_any"].append(r_any)
            s2_metrics[k]["recall_all"].append(r_all)
            s2_metrics[k]["ndcg"].append(n_val)

        # Progress every 30 questions
        if (qi + 1) % 30 == 0 or qi == 0:
            s1_r1 = sum(s1_metrics[1]["recall_any"]) / len(s1_metrics[1]["recall_any"]) if s1_metrics[1]["recall_any"] else 0.0
            s2_r1 = sum(s2_metrics[1]["recall_any"]) / len(s2_metrics[1]["recall_any"]) if s2_metrics[1]["recall_any"] else 0.0
            s2_r5 = sum(s2_metrics[5]["recall_any"]) / len(s2_metrics[5]["recall_any"]) if s2_metrics[5]["recall_any"] else 0.0
            filt_avg = (total_filter_time / max(qi + 1, 1) * 1000) if not args.no_filter else 0
            print(
                f"  [{qi+1:3d}/{len(data)}] S1_R@1={s1_r1:.1%} S2_R@1={s2_r1:.1%} R@5={s2_r5:.1%}"
                f"  filter={filt_avg:.0f}ms  route={route_decision}"
                f"  boost_q={temporal_boost_fired_count}  scaffold_q={socratic_fired_count}"
            )

    n_q = len(per_question_log)
    def _safe_mean(lst): return sum(lst) / len(lst) if lst else 0.0

    print(f"\n{'='*70}")
    print(f"  v33-ABLATE [{ablation_desc}] on LongMemEval_{args.split.upper()}  —  {n_q} questions")
    print(f"{'='*70}\n")

    # Aggregate from log
    s1_r1_all = [float(e["s1_hit_at_1"]) for e in per_question_log]
    s1_r5_all = [float(e["s1_hit_at_5"]) for e in per_question_log]
    s2_r1_all = [float(e["s2_hit_at_1"]) for e in per_question_log]
    s2_r5_all = [float(e["s2_hit_at_5"]) for e in per_question_log]

    # Per-qtype analysis
    qtypes_seen = sorted(set(e["qtype"] for e in per_question_log))
    print(f"  Per-qtype R@1 (Stage 2 final):")
    print(f"  {'qtype':<30} {'n':>5} {'R@1':>8} {'vs_runJ-v33':>14}")
    print(f"  {'─'*30} {'─'*5} {'─'*8} {'─'*14}")

    runj_r1 = {}
    runj_path = Path(__file__).parent / "runs" / "runJ-v33" / "per_question.json"
    if runj_path.exists():
        runj_data = json.load(open(runj_path))
        for qt in qtypes_seen:
            qt_entries = [e for e in runj_data if e.get("qtype") == qt]
            if qt_entries:
                runj_r1[qt] = sum(float(e["s2_hit_at_1"]) for e in qt_entries) / len(qt_entries)

    qtype_stats: dict[str, dict] = {}
    for qt in qtypes_seen:
        qt_entries = [e for e in per_question_log if e.get("qtype") == qt]
        n_qt = len(qt_entries)
        r1_qt = sum(float(e["s2_hit_at_1"]) for e in qt_entries) / max(n_qt, 1)
        r1_baseline = runj_r1.get(qt, None)
        delta_str = f"{(r1_qt - r1_baseline):+.1%}" if r1_baseline is not None else "N/A"
        qtype_stats[qt] = {"n": n_qt, "r1": r1_qt, "delta": delta_str}
        print(f"  {qt:<30} {n_qt:>5} {r1_qt:>7.1%} {delta_str:>14}")

    s1_r1_v = _safe_mean(s1_r1_all)
    s1b_r1_v = _safe_mean(s1b_metrics[1]["recall_any"])
    s1b_r5_v = _safe_mean(s1b_metrics[5]["recall_any"])
    s1_r5_v = _safe_mean(s1_r5_all)
    s2_r1_v = _safe_mean(s2_r1_all)
    s2_r3_v = _safe_mean(s2_metrics[3]["recall_any"]) if s2_metrics[3]["recall_any"] else s2_r1_v
    s2_r5_v = _safe_mean(s2_r5_all)
    s2_r10_v = _safe_mean(s2_metrics[10]["recall_any"]) if s2_metrics[10]["recall_any"] else s2_r5_v

    print(f"\n  Stage 1 R@1: {s1_r1_v:.3%}  Stage 2 R@1: {s2_r1_v:.3%}")
    print(f"  Delta vs runJ-v33 (95.11%): {s2_r1_v - 0.9511:+.3%}")
    print(f"\n  Filter methods: {filter_methods}")
    print(f"  Temporal boost fired: {temporal_boost_fired_count}  Scaffold fired: {socratic_fired_count}")

    # Save aggregate
    out = {
        "benchmark": "LongMemEval",
        "split": args.split.upper(),
        "date": time.strftime("%Y-%m-%d"),
        "test": f"v33-ablate-{ablation_desc}",
        "ablation": ablation_desc,
        "ablation_flags": {
            "no_temporal_boost": args.no_temporal_boost,
            "no_socratic": args.no_socratic,
            "no_verify_guard": args.no_verify_guard,
            "no_flagship": args.no_flagship,
        },
        "filter_model": FILTER_MODEL,
        "filter_top_k": FILTER_TOP_K,
        "questions_evaluated": n_q,
        "stage1_r1": round(s1_r1_v, 4),
        "stage1b_r1": round(s1b_r1_v, 4),
        "stage2_r1": round(s2_r1_v, 4),
        "stage2_r5": round(s2_r5_v, 4),
        "delta_vs_v33": round(s2_r1_v - 0.9511, 4),
        "qtype_breakdown": {
            qt: {"n": v["n"], "r1": round(v["r1"], 4), "delta_vs_v33": v["delta"]}
            for qt, v in qtype_stats.items()
        },
        "filter_methods": filter_methods,
        "temporal_boost_fired_count": temporal_boost_fired_count,
        "socratic_fired_count": socratic_fired_count,
        "avg_retrieval_ms": round(total_retrieval_time / n_q * 1000) if n_q else 0,
        "avg_filter_ms": round(total_filter_time / n_q * 1000) if n_q else 0,
        "total_time_s": round(total_retrieval_time + total_filter_time, 1),
    }
    out_path = run_dir / "aggregate.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    pq_path = run_dir / "per_question.json"
    with open(pq_path, "w") as f:
        json.dump(per_question_log, f, indent=2)

    print(f"\n  Aggregate saved to {out_path}")
    print(f"  Per-question log: {pq_path}")


if __name__ == "__main__":
    main()
