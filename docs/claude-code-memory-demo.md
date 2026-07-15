# Claude Code Session Memory — Demo

cogito-ergo v0.3.0 can ingest your Claude Code sessions and query them at 93%+ accuracy
using the same turn-pair chunking pipeline as the LongMemEval benchmark.

## Setup (one-time)

cogito-ergo is already running at localhost:19420.
The ingest tool is at `src/cogito/ingest_claude_sessions.py`.

## Step 1 — Preview what would be ingested

```bash
python3 -m cogito.ingest_claude_sessions --since 2026-04-11 --dry-run
```

Output:
```
[cogito-ingest] DRY RUN — scanning ~/.claude/projects
[cogito-ingest] Since: 2026-04-11
  DRY-RUN  b2fff5ab | 100 turns | -Users-rbr-lpci | "You are evaluating memory systems..."
  DRY-RUN  a7b08693 | 100 turns | -Users-rbr-lpci | "You are continuing cogito-ergo benchmark..."
  DRY-RUN  3d3f08d1 | 100 turns | -Users-rbr-lpci | "You are the Tournament Director..."
  ...
[cogito-ingest] Done.
  Scanned:       63
  Stored:        62
  Skipped empty: 1
```

## Step 2 — Ingest

```bash
python3 -m cogito.ingest_claude_sessions --since 2026-04-16
```

```
[cogito-ingest] LIVE — scanning ~/.claude/projects
[cogito-ingest] Since: 2026-04-16

[cogito-ingest] Done.
  Scanned:       32
  Stored:        32
  Skipped dedup: 0
```

## Step 3 — Query

```python
from cogito.recall_sessions import query_sessions

results = query_sessions("cogito session ingestion recall", top_k=3)
for r in results:
    print(f"session={r.session_id[:12]}  turns={r.turn_count}  date={r.start_ts[:10]}")
    print(f"  {r.matched_chunk[:300]}")
```

### Actual output (from real April 16-17 sessions):

```
session=b2fff5ab-2c7  turns=100  date=2026-04-16
  User: wait... what. cogito + langstate would be better than mem0?
  Assistant: Good challenge. Let me be precise about what mem0 actually is here,
  because cogito already uses it. **mem0 is inside cogito.** It's a dependency...

session=a7b08693-7f7  turns=100  date=2026-04-15
  User: You are continuing cogito-ergo benchmark work at ~/Documents/projects/cogito-ergo.
  Previous results: R@1 80%, R@any 94.4% on LongMemEval.
  Task: Run the benchmark, save results, stop.
```

## Step 4 — Via MCP tools (in Claude Code)

Two new tools are available in every Claude Code session:

**`cogito_recall_sessions`** — query session history only
```
Query: "what did we discuss about LPCI proof last week?"
→ Returns 3 matched sessions with turn-pair context
```

**`cogito_recall_both`** — atomic facts + session history side-by-side
```
=== ATOMIC (facts) ===
  [1] LPCI 2026-03-28: one 20-turn A/B run tested language-scaffold state; single observation, not proof
  ...

=== SESSIONS (Claude Code history) ===
  [1] session=b2fff5ab | score=0.8712 | turns=100 | 2026-04-16
      User: wait... what. cogito + langstate would be better than mem0?
```

## Re-ingestion is idempotent

Running ingest twice on the same sessions stores nothing the second time:

```bash
python3 -m cogito.ingest_claude_sessions --since 2026-04-16
# Stored: 0 | Skipped dedup: 32
```

## Privacy note

Sessions contain full conversation history. The ingestion script:
- Is **explicit** — nothing runs automatically
- Uses **local embedding only** (Ollama nomic-embed-text at localhost:11434)
- Stores data in **local ChromaDB** (~/.cogito/store)
- Sends **nothing to any cloud API** by default
- Dedup ledger at `~/.cogito/session_ingest/ingested.json`

If Ollama is not running: `ollama serve` (or it may already be running).
