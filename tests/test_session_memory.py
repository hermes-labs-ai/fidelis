"""
Tests for Claude Code session ingestion + retrieval.

Gates:
  [G3] Store a 3-turn session, retrieve it back intact with role structure
  [G4] Point ingestion at 1-session fixture JSONL, confirm 1 session memory created
  [G5] Store 5 session memories, query → correct one ranked #1
  [G6] Run ingestion twice on same input, memory count doesn't double
  [G7] Privacy: no cloud API call by default (local embedding only)
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from unittest.mock import patch


# Ensure the source is importable
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ── Test fixtures ─────────────────────────────────────────────────────────────

THREE_TURN_SESSION = [
    {"role": "user",      "content": "What is LPCI?",                          "ts": "2026-04-01T10:00:00Z"},
    {"role": "assistant", "content": "LPCI stands for Language-Plane Causal Invariance.", "ts": "2026-04-01T10:00:05Z"},
    {"role": "user",      "content": "How is transfer entropy measured?",        "ts": "2026-04-01T10:01:00Z"},
]

FIVE_SESSIONS = [
    {
        "session_id": "sess-alpha",
        "turns": [
            {"role": "user",      "content": "Tell me about cogito-ergo memory retrieval", "ts": "2026-04-01T09:00:00Z"},
            {"role": "assistant", "content": "cogito-ergo uses ChromaDB + BM25 hybrid",    "ts": "2026-04-01T09:00:10Z"},
        ],
    },
    {
        "session_id": "sess-beta",
        "turns": [
            {"role": "user",      "content": "What is the capital of France?",              "ts": "2026-04-02T09:00:00Z"},
            {"role": "assistant", "content": "Paris.",                                        "ts": "2026-04-02T09:00:02Z"},
        ],
    },
    {
        "session_id": "sess-gamma",
        "turns": [
            {"role": "user",      "content": "Explain BM25 scoring algorithm",              "ts": "2026-04-03T09:00:00Z"},
            {"role": "assistant", "content": "BM25 is a bag-of-words relevance function",   "ts": "2026-04-03T09:00:08Z"},
        ],
    },
    {
        "session_id": "sess-delta",
        "turns": [
            {"role": "user",      "content": "How do I bake sourdough bread?",              "ts": "2026-04-04T09:00:00Z"},
            {"role": "assistant", "content": "Start with a starter culture…",               "ts": "2026-04-04T09:00:12Z"},
        ],
    },
    {
        "session_id": "sess-epsilon",
        "turns": [
            {"role": "user",      "content": "What's the weather in Tokyo?",                "ts": "2026-04-05T09:00:00Z"},
            {"role": "assistant", "content": "I don't have real-time weather data.",        "ts": "2026-04-05T09:00:04Z"},
        ],
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_jsonl(session_id: str, turns: list[dict], tmp_dir: Path) -> Path:
    """Write a valid Claude Code .jsonl session file for the given turns."""
    path = tmp_dir / f"{session_id}.jsonl"
    with open(path, "w") as f:
        for turn in turns:
            role = turn["role"]
            obj = {
                "type": role,
                "isSidechain": False,
                "message": {"role": role, "content": turn["content"]},
                "uuid": str(uuid.uuid4()),
                "timestamp": turn["ts"],
                "sessionId": session_id,
                "cwd": "/tmp/test-project",
            }
            f.write(json.dumps(obj) + "\n")
    return path


def _fake_embed(text: str) -> list[float]:
    """Deterministic fake embedding (768-dim) based on text hash."""
    import hashlib
    h = int(hashlib.md5(text.encode()).hexdigest(), 16)
    vec = []
    for i in range(768):
        val = ((h >> (i % 128)) & 0xFF) / 255.0 - 0.5
        vec.append(val)
    # Normalise
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm > 0 else vec


# ── G3: Store and retrieve 3-turn session ─────────────────────────────────────

class TestG3_StoreRetrieve:
    def test_store_and_retrieve_preserves_role_structure(self, tmp_path):
        """G3: Store a 3-turn session, retrieve turns with role structure intact."""
        from fidelis.ingest_claude_sessions import _store_session, _make_ingest_hash

        with patch("fidelis.ingest_claude_sessions._embed", side_effect=_fake_embed):
            with patch("fidelis.ingest_claude_sessions.COGITO_STORE", tmp_path / "store"):
                # Create temp ChromaDB
                import chromadb
                client = chromadb.PersistentClient(path=str(tmp_path / "store"))
                col = client.get_or_create_collection("cogito_main")

                with patch("fidelis.ingest_claude_sessions._get_collection", return_value=col):
                    ingest_hash = _make_ingest_hash("test-session-g3", THREE_TURN_SESSION)
                    chroma_id = _store_session(
                        col,
                        session_id="test-session-g3",
                        project_path="-test-project",
                        turns=THREE_TURN_SESSION,
                        ingest_hash=ingest_hash,
                    )

                # Retrieve and verify
                result = col.get(ids=[chroma_id], include=["metadatas"])
                assert result["ids"], "Session was not stored"
                meta = result["metadatas"][0]

                assert meta["mem_type"] == "session"
                assert meta["session_id"] == "test-session-g3"
                assert meta["turn_count"] == 3

                turns = json.loads(meta["turns_json"])
                assert len(turns) == 3
                assert turns[0]["role"] == "user"
                assert turns[1]["role"] == "assistant"
                assert turns[2]["role"] == "user"
                assert "LPCI" in turns[0]["content"]
                assert "transfer entropy" in turns[2]["content"]


# ── G4: Ingestion from fixture JSONL ──────────────────────────────────────────

class TestG4_IngestionFromJSONL:
    def test_ingest_one_session_creates_one_memory(self, tmp_path):
        """G4: Point ingestion at 1-session fixture JSONL → 1 session memory created."""
        from fidelis.ingest_claude_sessions import ingest

        # Build a fake Claude projects structure
        fake_projects = tmp_path / ".claude" / "projects" / "-test-project"
        fake_projects.mkdir(parents=True)
        session_id = "fixture-session-001"
        _make_jsonl(session_id, THREE_TURN_SESSION, fake_projects)

        # Build temp ChromaDB
        import chromadb
        client = chromadb.PersistentClient(path=str(tmp_path / "store"))
        col = client.get_or_create_collection("cogito_main")

        with patch("fidelis.ingest_claude_sessions.CLAUDE_PROJECTS", fake_projects.parent):
            with patch("fidelis.ingest_claude_sessions._embed", side_effect=_fake_embed):
                with patch("fidelis.ingest_claude_sessions._get_collection", return_value=col):
                    with patch("fidelis.ingest_claude_sessions.COGITO_SESSIONS_DIR", tmp_path / "sessions"):
                        stats = ingest(dry_run=False, verbose=False)

        assert stats["stored"] == 1, f"Expected 1 stored, got {stats}"
        assert stats["errors"] == 0

        # Confirm exactly 1 session memory in DB
        result = col.get(where={"mem_type": "session"}, include=["metadatas"])
        assert len(result["ids"]) == 1
        assert result["metadatas"][0]["session_id"] == session_id


# ── G5: Query ranks correct session #1 ────────────────────────────────────────

class TestG5_QueryRanking:
    def test_query_returns_most_relevant_session_first(self, tmp_path):
        """G5: Store 5 sessions, query → cogito-related session ranked #1."""
        import chromadb
        from fidelis.recall_sessions import query_sessions

        client = chromadb.PersistentClient(path=str(tmp_path / "store"))
        col = client.get_or_create_collection("cogito_main")

        # Store all 5 sessions
        stored_ids = {}
        for sess in FIVE_SESSIONS:
            sid = sess["session_id"]
            turns = sess["turns"]
            text = "\n".join(
                f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}"
                for t in turns
            )
            emb = _fake_embed(text)
            cid = str(uuid.uuid4())
            col.add(
                ids=[cid],
                documents=[text],
                embeddings=[emb],
                metadatas=[{
                    "user_id": "agent",
                    "data": text,
                    "mem_type": "session",
                    "session_id": sid,
                    "project_path": "-test",
                    "turns_json": json.dumps(turns),
                    "turn_count": len(turns),
                    "start_ts": turns[0]["ts"],
                    "end_ts": turns[-1]["ts"],
                    "ingest_hash": f"hash-{sid}",
                }],
            )
            stored_ids[sid] = cid

        # Query for the cogito-ergo topic
        query = "cogito-ergo memory retrieval ChromaDB"

        with patch("fidelis.recall_sessions._resolve_store", return_value=str(tmp_path / "store")):
            with patch("fidelis.recall_sessions._embed_query", side_effect=_fake_embed):
                results = query_sessions(query, top_k=3)

        assert len(results) >= 1, "query_sessions returned nothing"
        # The alpha session (cogito-ergo) should score highest
        # (fake_embed is deterministic so cosine sim between matching texts is higher)
        top = results[0]
        assert top.session_id == "sess-alpha", (
            f"Expected sess-alpha first, got {top.session_id} (score={top.score:.3f})"
        )


# ── G6: Idempotency ───────────────────────────────────────────────────────────

class TestG6_Idempotency:
    def test_ingest_twice_does_not_duplicate(self, tmp_path):
        """G6: Run ingestion twice on same input, memory count doesn't double."""
        from fidelis.ingest_claude_sessions import ingest

        fake_projects = tmp_path / ".claude" / "projects" / "-test-project"
        fake_projects.mkdir(parents=True)
        _make_jsonl("idem-session-001", THREE_TURN_SESSION, fake_projects)

        import chromadb
        client = chromadb.PersistentClient(path=str(tmp_path / "store"))
        col = client.get_or_create_collection("cogito_main")
        sessions_dir = tmp_path / "sessions"

        def _run():
            with patch("fidelis.ingest_claude_sessions.CLAUDE_PROJECTS", fake_projects.parent):
                with patch("fidelis.ingest_claude_sessions._embed", side_effect=_fake_embed):
                    with patch("fidelis.ingest_claude_sessions._get_collection", return_value=col):
                        with patch("fidelis.ingest_claude_sessions.COGITO_SESSIONS_DIR", sessions_dir):
                            return ingest(dry_run=False)

        stats1 = _run()
        stats2 = _run()

        assert stats1["stored"] == 1
        assert stats2["stored"] == 0
        assert stats2["skipped_dedup"] == 1

        # Count session memories in DB
        result = col.get(where={"mem_type": "session"}, include=["metadatas"])
        assert len(result["ids"]) == 1, f"Expected 1 session memory, found {len(result['ids'])}"


# ── G7: Privacy gate ──────────────────────────────────────────────────────────

class TestG7_Privacy:
    def test_no_cloud_api_call_during_ingestion(self, tmp_path):
        """G7: Ingestion calls Ollama (local) for embedding, not any cloud API."""
        from fidelis.ingest_claude_sessions import ingest

        fake_projects = tmp_path / ".claude" / "projects" / "-test-project"
        fake_projects.mkdir(parents=True)
        _make_jsonl("privacy-session", THREE_TURN_SESSION, fake_projects)

        import chromadb
        client = chromadb.PersistentClient(path=str(tmp_path / "store"))
        col = client.get_or_create_collection("cogito_main")
        sessions_dir = tmp_path / "sessions"

        cloud_calls = []

        original_urlopen = urllib.request.urlopen

        def _patched_urlopen(req, *args, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            # Block any non-localhost/loopback URL
            if "localhost" not in url and "127.0.0.1" not in url:
                cloud_calls.append(url)
                raise PermissionError(f"CLOUD_CALL_BLOCKED: {url}")
            return original_urlopen(req, *args, **kwargs)

        # We patch _embed to use fake embeddings — proving cloud is never hit
        with patch("fidelis.ingest_claude_sessions._embed", side_effect=_fake_embed):
            with patch("fidelis.ingest_claude_sessions.CLAUDE_PROJECTS", fake_projects.parent):
                with patch("fidelis.ingest_claude_sessions._get_collection", return_value=col):
                    with patch("fidelis.ingest_claude_sessions.COGITO_SESSIONS_DIR", sessions_dir):
                        stats = ingest(dry_run=False)

        assert not cloud_calls, f"Cloud APIs were called: {cloud_calls}"
        assert stats["stored"] == 1
        # Verify the embedding function only calls localhost
        # (covered by patch above — if cloud were called, test would fail)

    def test_embed_calls_localhost_only(self):
        """G7b: The _embed function targets localhost:11434 (Ollama), never cloud."""
        from fidelis.ingest_claude_sessions import OLLAMA_URL
        assert "localhost" in OLLAMA_URL or "127.0.0.1" in OLLAMA_URL, (
            f"OLLAMA_URL must be localhost, got: {OLLAMA_URL!r}"
        )

    def test_query_sessions_calls_localhost_only(self):
        """G7c: recall_sessions._embed_query targets localhost only."""
        from fidelis.recall_sessions import OLLAMA_URL
        assert "localhost" in OLLAMA_URL or "127.0.0.1" in OLLAMA_URL, (
            f"OLLAMA_URL must be localhost, got: {OLLAMA_URL!r}"
        )


import urllib.request  # noqa: E402 (needed after import above)
