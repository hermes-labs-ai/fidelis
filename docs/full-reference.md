# Fidelis Memory 0.3.0rc1 reference

## Install and run

```bash
python3 -m pip install "fidelis-memory==0.3.0rc1"
ollama pull nomic-embed-text
fidelis init
fidelis mcp install --client codex
```

Use `fidelis-server` for foreground operation. Default HTTP address:
`http://127.0.0.1:19420`. Both client and server honor `FIDELIS_PORT` before the
legacy `COGITO_PORT`. The MCP stdio command is `fidelis mcp serve` (or
`fidelis-mcp`). It discovers tools without loading the memory store.

## Record model

A record contains original text and a stable ID. The service sets `recorded_at`;
the caller may declare `valid_from`, `valid_to`, `event_at`, `source`, and
`supersedes` IDs. Corrections create new records. Earlier records remain available
and receive temporal status such as `superseded`, `expired`, or `not_yet_valid`.
Declared dates and links are not independent verification of truth.

`as_of` excludes records with a known `recorded_at` later than the requested
instant, ignores later correction links, and evaluates validity at that instant.
Backdating validity does not backdate recording time. Legacy records with unknown
recording times remain visible and are labeled unknown; they cannot establish
what the system knew at a particular earlier time.

## HTTP endpoints

JSON POST requests use `Content-Type: application/json`.

| Endpoint | Request | Behavior |
|---|---|---|
| `GET /health` | — | Version, availability, store/queue state; does not require eager model boot |
| `POST /store` | `{"text":"fact"}` | Verbatim write; optional temporal fields, ID and metadata |
| `POST /query` | `{"text":"question","limit":5}` | Fast local vector search with temporal presentation |
| `POST /get` | `{"id":"record-id"}` | Full record and correction chain |
| `POST /recent` | `{"limit":10,"kind":"all"}` | Latest recorded entries; `kind:"corrections"` selects replacements |
| `POST /recall_hybrid` | `{"text":"question","limit":5,"top_k":5,"tier":"zero_llm"}` | Explicit hybrid retrieval |
| `POST /recall_b` | `{"text":"question","limit":5}` | Legacy decomposed zero-LLM retrieval |
| `POST /recall` | `{"text":"question","limit":5}` | Legacy retrieval with optional configured LLM filtering |
| `POST /add` | `{"text":"raw material"}` | Optional model extraction; separate from verbatim store |
| `POST /replay` | `{}` | Request queued-write replay |

Retrieval requests accept `as_of` as an ISO-8601 timestamp. The normal current
view retains historical records with explicit labels; callers should inspect
status, not assume every hit is current. Limits bound the result count.

Example correction through HTTP:

```json
{"text":"The Atlas release window starts at 10:00 UTC.","supersedes":["earlier-id"],"valid_from":"2026-02-01T00:00:00Z"}
```

The `supersedes` target must exist in the configured namespace. HTTP clients
should check both status codes and response bodies. Memory operations can return
503 when the backing store cannot load; health/discovery remain available.

## Read-only CLI browsing

The CLI exposes the same bounded `/recent` and point-lookup `/get` paths. These
commands require a running `fidelis-server` and do not load or scan the store in
the client process.

```bash
fidelis recent
fidelis recent --limit 20 --kind corrections --since 2026-09-01T00:00:00Z
fidelis get RECORD_ID
```

`recent` accepts a limit from 1 to 50 and `kind` as `all` or `corrections`.
Human-readable output includes stable IDs, record text, timestamps, temporal
status, sources, and correction links when present. Add `--raw` to either
command to print the service JSON payload for scripting. An unavailable service,
invalid request, or unknown record exits nonzero with an explicit error.

## Write acknowledgements

- `stored`: the write landed.
- `duplicate`: an identical existing record was found; no new record was written.
- `rejected`: screening or validation refused the write.
- `queued`: the write is durably awaiting replay; it has not landed in the store.

Temporal declarations matter to identity: repeating text with different validity
or correction declarations is not necessarily an identical write. A rejected
secret must not be treated as safely retained elsewhere. Screening recognizes
common secret patterns and obvious noise, not all sensitive information.

Optional `/add` extraction may preserve input verbatim if extraction returns no
facts. Read `degraded` in the response; successful storage does not prove that
model extraction succeeded.

## MCP contracts

Exactly six tools are exposed: `fidelis_recall`, `fidelis_store`,
`fidelis_correct`, `fidelis_get`, `fidelis_recent`, `fidelis_health`.

- Recall requires `query`, defaults to `limit:5`, and accepts `mode:"fast"` or
  `mode:"thorough"`. Fast calls `/query`; thorough calls `/recall_hybrid` with
  `tier:"zero_llm"`. It does not silently escalate to a generative model.
- Store accepts `text` and optional temporal/source metadata.
- Correct takes the old `id` and replacement `text`. Already-superseded IDs are
  refused unless the caller explicitly chooses `force:true`.
- Get takes `id` and returns the full record plus correction links.
- Recent accepts `limit`, `kind` (`all` or `corrections`), and optional `since`.
- Health reports availability; a lazy, unloaded store is not an empty corpus.

The old four-tool surface and `cogito_*` compatibility tool names are retired.
Reconfigure standing instructions and restart MCP clients on upgrade.
JSON-RPC batches are accepted only when the negotiated protocol supports them
(`2025-03-26`); malformed input must not terminate the stdio service.

## Retrieval architecture

The default MCP path is a bounded vector search, followed by temporal/status
handling. It uses the local `nomic-embed-text` model; zero-LLM means no generative
model, not an absence of learned embeddings or local compute.

Thorough retrieval assembles candidates using multiple dense subqueries and,
with the `hybrid` extra, applies BM25 within that candidate pool. It merges
rankings through reciprocal rank fusion; BM25 does not independently search
the entire corpus. Without
BM25 the dense path remains available. Install the extra with:

```bash
python3 -m pip install "fidelis-memory[hybrid]==0.3.0rc1"
```

Legacy optional filter/flagship tiers can call configured model endpoints. They
are not the default MCP modes. Historical benchmarks of those pipelines must
not be attributed to the redesigned fast path. Full QA evaluation is deferred.

## Local configuration and boundaries

Environment variables override file configuration. `fidelis-server --config PATH`
selects an explicit JSON config; otherwise `.cogito.json` then
`~/.cogito/config.json` are searched.

- `COGITO_STORE_PATH`: vector store directory, default `~/.cogito/store`.
- `COGITO_COLLECTION`: Chroma collection.
- `COGITO_USER_ID`: local namespace, not authentication.
- `FIDELIS_QUEUE_DIR` / `COGITO_QUEUE_DIR`: queue directory.
- `COGITO_OLLAMA_URL`: embedding/model service endpoint.
- `COGITO_EMBED_MODEL`: embedding model; keep one model per collection.
- `FIDELIS_RETRIEVAL_TELEMETRY_LOG`: optional retrieval diagnostics location.

Never expose the local HTTP service to untrusted networks. Local data is not
application-encrypted. Back up the entire configured store and queue before an
upgrade; preserve the old environment for rollback. See [SECURITY.md](../SECURITY.md).

## Native Gemini extension

Alternatively, install the release-pinned native extension:

```sh
gemini extensions install https://github.com/hermes-labs-ai/fidelis --ref=v0.3.0rc1
```

`gemini-extension.json` pins the same PyPI package as the MCP registry manifest.
