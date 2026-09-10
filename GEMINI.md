# Fidelis Memory

This extension connects Gemini CLI to a Fidelis Memory server running on this
machine. Fidelis stores the user's notes and decisions verbatim and retrieves
them without an LLM. The tools below return the original passages, not
summaries.

## When to use the tools

- `fidelis_orient` — call first when the user refers to prior work, a past
  decision, an earlier failure, an ongoing project, or something they "said"
  or "decided" before, even if the turn is not phrased as a question. It
  returns an evidence-bound orientation packet or explicitly abstains.
- `fidelis_recall` — retrieve the passages that answer a specific question
  about what was recorded. Quote qualifiers exactly as returned.
- `fidelis_query` — fast vector-only lookup when a short candidate list is
  enough.
- `fidelis_health` — check that the local server is reachable and how many
  memories it holds. Use it when another tool reports the server as
  unreachable.

## Rules

- Ground answers about prior decisions in retrieved passages. Do not
  paraphrase a constraint or number that the retrieved text states exactly.
- If the tools report `fidelis-server unreachable`, tell the user to start it
  with `fidelis init` (first run) or `fidelis-server` and do not invent memory.
- Unrelated turns do not need these tools.

Requirements and the full command reference live in the repository README:
https://github.com/hermes-labs-ai/fidelis
