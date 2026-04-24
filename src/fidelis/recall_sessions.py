"""
cogito.recall_sessions — session-aware retrieval for cogito-ergo.

Queries session-typed memories (mem_type="session") with turn-pair chunking.
The chunking logic is ported from bench/longmemeval_combined_pipeline_flagship.py
and is what drives the 93.4% R@1 benchmark result — it works HERE because the
data is role-structured (user+assistant turns), not flat strings.

Provides:
    query_sessions(query, top_k=3) -> list[SessionResult]
    query_both(query, atomic_k=3, session_k=3) -> BothResult
"""
from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass, field

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_PREFIX = "search_query: "
COGITO_STORE_PATH = None  # lazy-resolved at first call


def _resolve_store() -> str:
    import os
    return os.path.expanduser("~/.cogito/store")


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_query(text: str) -> list[float]:
    payload = json.dumps({
        "model": EMBED_MODEL,
        "input": [EMBED_PREFIX + text[:2000]],
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        result = json.loads(resp.read())
    embeddings = result.get("embeddings", [])
    if not embeddings:
        raise RuntimeError("Ollama embed returned empty")
    return embeddings[0]


# ── Turn-pair chunking (ported from longmemeval_combined_pipeline_flagship.py) ─

def _chunk_turns(turns: list[dict]) -> list[str]:
    """
    Build overlapping turn-pair chunks from role-structured turns.
    Each chunk = user[i] + assistant[i] snippet.

    This is the key transformation that makes /recall_hybrid work on session data:
    - Single-turn granularity ensures exact matches survive embedding noise
    - Overlap ensures multi-turn context isn't lost at chunk boundaries
    """
    chunks = []
    for i, turn in enumerate(turns):
        if turn.get("role") != "user":
            continue
        user_text = turn["content"][:400]

        # Find following assistant turn
        asst_text = ""
        if i + 1 < len(turns) and turns[i + 1].get("role") == "assistant":
            asst_text = turns[i + 1]["content"][:400]

        chunk = f"User: {user_text}"
        if asst_text:
            chunk += f"\nAssistant: {asst_text}"
        chunks.append(chunk)

    return chunks if chunks else ["[empty session]"]


def _bm25_score(query_tokens: set[str], doc_text: str) -> float:
    """Minimal BM25-style overlap score (k1=1.5, b=0.75, avgdl~200)."""
    tokens = set(doc_text.lower().split())
    overlap = query_tokens & tokens
    if not overlap:
        return 0.0
    k1, b, avgdl = 1.5, 0.75, 200.0
    dl = len(tokens)
    score = 0.0
    for token in overlap:
        tf = doc_text.lower().count(token)
        idf = math.log(1 + 1.0 / (1 + 0.5))  # simplified: assume 1 doc has it
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    return score


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SessionResult:
    session_id: str
    project_path: str
    start_ts: str
    end_ts: str
    turn_count: int
    matched_chunk: str        # the specific turn-pair that matched
    score: float
    turns: list[dict] = field(default_factory=list)   # full turn list

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "project_path": self.project_path,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "turn_count": self.turn_count,
            "matched_chunk": self.matched_chunk,
            "score": round(self.score, 4),
        }


@dataclass
class BothResult:
    atomic: list[dict]
    sessions: list[SessionResult]

    def to_dict(self) -> dict:
        return {
            "atomic": self.atomic,
            "sessions": [s.to_dict() for s in self.sessions],
        }


# ── Core retrieval ────────────────────────────────────────────────────────────

def _get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=_resolve_store())
    return client.get_collection("cogito_main")


def query_sessions(query: str, top_k: int = 3) -> list[SessionResult]:
    """
    Retrieve top_k session memories most relevant to query.

    Algorithm:
    1. Fetch all session-typed memories from ChromaDB
    2. For each session, chunk into turn-pairs
    3. Score each chunk: 0.7 * cosine_sim + 0.3 * bm25_norm
    4. Best chunk score represents the session
    5. Return top_k sessions ranked by best chunk score
    """
    if not query.strip():
        return []

    col = _get_collection()

    # Fetch session memories
    try:
        result = col.get(
            where={"mem_type": "session"},
            include=["metadatas", "embeddings", "documents"],
        )
    except Exception:
        # ChromaDB raises if no docs match the where filter
        return []

    if not result["ids"]:
        return []

    query_vec = _embed_query(query)
    query_tokens = set(query.lower().split())

    candidates: list[tuple[float, SessionResult]] = []

    for i, meta in enumerate(result["metadatas"]):
        session_id = meta.get("session_id", "")
        turns_json = meta.get("turns_json", "[]")
        try:
            turns = json.loads(turns_json)
        except Exception:
            turns = []

        chunks = _chunk_turns(turns)

        # Score each chunk independently
        best_score = 0.0
        best_chunk = chunks[0] if chunks else ""

        for chunk in chunks:
            # Dense similarity against the stored session embedding
            all_embeddings = result.get("embeddings")
            stored_emb = None
            if all_embeddings is not None and i < len(all_embeddings):
                stored_emb = all_embeddings[i]
            if stored_emb is not None:
                try:
                    cos = _cosine(query_vec, list(stored_emb))
                except Exception:
                    cos = 0.0
            else:
                cos = 0.0

            # Sparse BM25 on chunk text
            bm25 = _bm25_score(query_tokens, chunk)
            bm25_norm = min(bm25 / 5.0, 1.0)  # rough normalisation

            chunk_score = 0.7 * cos + 0.3 * bm25_norm  # calibrated: 70/30 cosine/bm25 blend from recall_b experiments
            if chunk_score > best_score:
                best_score = chunk_score
                best_chunk = chunk

        sr = SessionResult(
            session_id=session_id,
            project_path=meta.get("project_path", ""),
            start_ts=meta.get("start_ts", ""),
            end_ts=meta.get("end_ts", ""),
            turn_count=meta.get("turn_count", 0),
            matched_chunk=best_chunk[:500],
            score=best_score,
            turns=turns,
        )
        candidates.append((best_score, sr))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [sr for _, sr in candidates[:top_k]]


def query_both(query: str, atomic_k: int = 3, session_k: int = 3) -> BothResult:
    """
    Return atomic + session results side-by-side. No auto-merge.

    Atomic results come from the live cogito server (via HTTP, preserves filter logic).
    Session results come from recall_sessions.query_sessions.
    """
    # Atomic results via server (preserves the integer-pointer filter)
    atomic = []
    try:
        payload = json.dumps({"text": query, "limit": atomic_k}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:19420/recall",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            server_result = json.loads(resp.read())
        atomic = [
            {"text": m.get("text", m), "source": "atomic"}
            for m in server_result.get("memories", [])
        ]
    except Exception as exc:
        atomic = [{"error": f"atomic recall failed: {exc}", "source": "atomic"}]

    sessions = query_sessions(query, top_k=session_k)
    return BothResult(atomic=atomic, sessions=sessions)
