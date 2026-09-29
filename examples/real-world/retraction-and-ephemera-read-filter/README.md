# Read-time retraction annotation and ephemera filtering

## Job
An operator keeps a machine-readable pointers file. Recall hits that match a retracted claim
are annotated (never dropped), and hits that are raw tool or scaffold output can be hidden or
flagged.

## Files
- `pointers.json`: one `retracted` rule and three `ephemera_patterns`.
- `input.json`: seven hits.
- `expected.json`: results of `annotate_superseded`, `mark_superseded`, `filter_ephemera`,
  and `mark_ephemera`.

## How it is verified (no server)
The test copies `pointers.json` to a pytest temp directory, resets the module-level rule cache
in `fidelis.supersession`, points `cfg["supersession_pointers_path"]` at the temp file, and
calls the four functions. They read one JSON file and apply regexes; there is no network or store.

## What you should see
- `h1` (the old port claim) is kept and prefixed `[SUPERSEDED:gw-port-1] ... || original: ...`
  by `annotate_superseded`. `mark_superseded` instead leaves the text byte-identical and adds
  structured `supersession` metadata.
- `h3`, `h4`, `h5` are ephemera. `h4` only matches because the `User:` / `Assistant:`
  storage envelope is peeled per line before the anchored pattern is tried.
- `h6` and `h7` talk about QA prompts and yes/no answers in prose and are kept.

## Provenance
The rule keys (`pattern`, `status`, `note`, `id`, `ephemera_patterns`) come from
`src/fidelis/supersession.py`. Hit texts come from `tests/test_superseded_not_dropped.py` and
`tests/test_write_gate.py`. The three ephemera patterns paraphrase anchors named in the
`supersession.py` comments (`^\s*You are`, `^\s*\[?tool_result`, a "user is currently" form).
The retraction rule, its ID, and its note are invented. The operator's real pointers file is
not part of the repository and is not reproduced here.

## Limits
- Patterns are regexes. They are illustrative and will miss junk they do not anticipate and can
  hide real notes that happen to match. `filter_ephemera` drops hits; use `mark_ephemera` when
  you need to keep provenance.
- Both functions fail open: a missing or corrupt pointers file leaves hits untouched.
- A retraction rule matches text, not record identity. It says nothing about whether the
  retraction is correct. Record-level correction with `supersedes` is the separate example
  `correction-chain-temporal`.
- `annotate_superseded` rewrites the returned text; prefer `mark_superseded` when the original
  wording must stay verbatim.
