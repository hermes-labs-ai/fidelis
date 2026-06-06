# Security & Privacy

This document covers how fidelis handles your data, with specific detail on the
`fidelis sessions` feature, which indexes your Claude Code conversation history.

## Reporting a vulnerability

If you find a security issue, please open a private report via GitHub Security
Advisories on the repository, or email the maintainers. Please do not file public
issues for vulnerabilities.

## What `fidelis sessions` stores, and where

`fidelis sessions ingest` reads your Claude Code session files
(`~/.claude/projects/*/SESSION_ID.jsonl`) and writes a searchable copy into a
**local** ChromaDB store at `~/.cogito/store`.

Each indexed session is stored with:

- the **full conversation text** of the session's turns, retained in the
  `turns_json` metadata field (user and assistant turns, up to 100 turns per
  session);
- a flat text representation used as the embedding document and for retrieval
  display;
- metadata: session id, project path, turn count, and timestamps.

There are no cloud calls in the default path: embeddings are produced locally by
Ollama (`nomic-embed-text`) and stored locally by ChromaDB. Nothing is uploaded.

A dedup ledger and purge log live alongside the store at
`~/.cogito/session_ingest/`.

## How the stored data is protected

The session text is **not application-encrypted at rest.** It is plain text
inside a local ChromaDB store. Protection relies on:

- **Filesystem permissions.** Ingest sets `chmod 700` on `~/.cogito`, so the
  directory is readable only by your OS user.
- **Full-disk encryption (FileVault on macOS, or your platform's equivalent).**
  Enable it. It is your at-rest encryption for this data.

If you share an account, run on a multi-user machine, or do not have full-disk
encryption, treat the contents of `~/.cogito` as readable plain text.

## Keeping sensitive projects out (pre-ingest lever)

The honest privacy lever is to **not index** sensitive projects in the first
place. Set `FIDELIS_SESSIONS_EXCLUDE` to a comma-separated list of substrings;
any project whose path contains one of them is never read or indexed:

```bash
export FIDELIS_SESSIONS_EXCLUDE="client-acme,secrets"
fidelis sessions ingest --all
```

This is also documented in `fidelis sessions ingest --help`.

## Deletion stops at the index

`fidelis sessions purge` removes matched sessions from the search index — their
text, embeddings, and metadata records — and cleans the dedup ledger entry.

Purge **never** touches your original Claude Code session files in
`~/.claude/projects` or any backups in `~/Backups/claude-sessions`. Those are
left alone by design. Purge prompts for confirmation unless you pass `--yes`.

To remove the entire store:

```bash
rm -rf ~/.cogito ~/.fidelis
```
