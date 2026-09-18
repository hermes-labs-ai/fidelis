---
name: fidelis-memory
description: Use when a turn refers to earlier work, a past decision, a named project, or prior context that is not in the transcript — recall it from the local fidelis store before answering, and write the decision back after the turn settles. Also use when asked to check what is already known about something, or to record a durable fact. Requires a running fidelis-server; the plugin's bundled MCP server provides fidelis_orient, fidelis_recall, fidelis_query, and fidelis_health.
license: MIT
---

# fidelis memory workflow

This skill is the procedure for using the `fidelis` MCP tools that ship with
this plugin. It does not describe how fidelis is built; see
<https://github.com/hermes-labs-ai/fidelis> for that.

The retrieval path is zero-LLM by default: the server returns stored passages
verbatim and never regenerates them. Treat what comes back as a quotation, not
as a paraphrase you may polish.

## Before you start

Call `fidelis_health` once per session. It is the only cheap way to tell the
three failure modes apart:

| Result | Meaning | What to do |
|---|---|---|
| `status: ok \| memories: N` with N > 0 | Store is live and populated | Proceed |
| `status: ok \| memories: 0` | Server is up, store is empty | Say so; do not present an empty recall as "nothing was decided" |
| `fidelis-server unreachable at …` | No server on the configured port | Stop using these tools and tell the user to run `fidelis init` (or `fidelis server`) |

An unreachable server is not an empty memory. Never report the two the same way.

## Recall before answering

When the turn leans on something outside the transcript, retrieve first.

1. `fidelis_orient` — the default for a conversational turn. Pass the user's
   utterance verbatim as `utterance`, plus up to four `recent_turns` when the
   referent is a pronoun ("that project", "the thing we changed"). It returns
   an evidence-bound orientation packet or explicitly abstains. **An abstention
   is an answer**: it means the store has nothing bound to that referent. Say
   that instead of guessing.
2. `fidelis_recall` — when you already know what you are looking for and want
   ranked passages. Two-stage; zero-LLM unless a filter endpoint is configured.
3. `fidelis_query` — a narrow vector-only lookup. Use it to confirm whether one
   specific string is present, not to survey a topic.

Quote recalled text as stored. If a recalled memory contradicts the current
turn, surface the contradiction rather than silently preferring either side —
the store records what was decided, not what is still true.

## Store after a decision

Write back when the turn produced something a later session would have to
re-derive: a decision and its reason, a constraint, a rejected option and why
it was rejected, a coordinate (a path, an ID, a version) that was expensive to
find.

Writes go over the local HTTP API, which the MCP surface does not expose — it
is read-only by design:

```bash
curl -s -X POST http://127.0.0.1:19420/store \
  -H 'Content-Type: application/json' \
  -d '{"text": "Chose uvx over a repo checkout for the plugin MCP command so install needs no clone. 2026-09-18."}'
```

Write one self-contained fact per call, in past tense, with the date. A memory
that only makes sense next to the message that produced it is not worth
storing — the retrieval path returns the passage alone.

Do not store secrets, credentials, or anything the user marked as sensitive.
The store is plaintext on local disk.

## Don't

- Don't call `fidelis_recall` on every turn. Recall when the turn reaches
  outside the transcript; skip it for self-contained work.
- Don't rewrite recalled text into your own words and present it as stored.
- Don't treat a score as confidence. It ranks candidates against each other
  within one query; it is not calibrated across queries.
- Don't assume `/store` reached the vector store. When the embedder is
  unreachable the server answers `{"status": "queued"}` and replays later —
  read the response.

## Configuration

`FIDELIS_PORT` (client side) points the MCP server at a non-default
fidelis-server port. Set it in this plugin's `.mcp.json` `env` block and start
the server on the matching port.
