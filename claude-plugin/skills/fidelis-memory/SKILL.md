---
name: fidelis-memory
description: Recall local records when a turn depends on prior work or decisions, or store and correct facts the user intends to retain. Requires a running Fidelis service; provides fidelis_recall, fidelis_store, fidelis_correct, fidelis_get, fidelis_recent, and fidelis_health.
license: Apache-2.0
---

# Fidelis memory workflow

Use `fidelis_recall` when the current turn needs earlier evidence. Default fast
recall uses local embeddings without a generative LLM. Use `as_of` for historical
validity and `mode: "thorough"` for explicit hybrid retrieval.

Read the record ID and temporal status alongside the text. Superseded records
remain visible as history. Quote the original wording and preserve constraints;
retrieval does not authenticate a claim or prove completeness. Similarity is a
ranking signal, not calibrated confidence. Empty results do not prove absence.

Use `fidelis_get` to fetch full text and correction links by ID. Use
`fidelis_recent` to browse recent records or corrections. On errors, inspect
`fidelis_health`; an unavailable or unloaded store is not an empty one. Ask the
user to start the local service and Ollama when needed.

When the user intends a durable fact to be kept, call `fidelis_store`. To replace
an earlier statement, call `fidelis_correct` with its ID and the replacement
text. The original remains in history. Optional validity dates describe when
the statement applies. Read the acknowledgement: queued is not stored, duplicate
creates no new fact, and rejected writes were not accepted. Do not invent success.

Do not store secrets or content the user has excluded. Storage is local and not
application-encrypted. Do not call recall for self-contained turns.

`FIDELIS_PORT` must match on the service and MCP process. Restart the client after
upgrading to refresh its six-tool surface.
