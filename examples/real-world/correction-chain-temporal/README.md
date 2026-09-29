# Correction chain: original, correction, supersession

## Job
A service fact changes. The old record must stay available as history, the new record must be
presented as current, and a caller replaying "what did we know on 5 March" must see the world
before the correction existed.

## Files
- `input.json`: three records (an original, an unrelated current record, and a correction that
  declares `supersedes: ["rec-a"]`), a relevance order, and two views.
- `expected.json`: the annotated hits each view produces.

## How it is verified (no server)
The test stamps each record with `fidelis.temporal.build_temporal_fields`, builds the
`SupersessionIndex` from the declared `supersedes` links, and runs
`fidelis.temporal.apply_temporal`. Those functions are standard-library only (see the module
docstring in `src/fidelis/temporal.py`) and touch no store.

## What you should see
- Current view: the correction (`rec-b`) and the unrelated record are `current`; the original
  (`rec-a`) is still returned, demoted to last and labeled `superseded` with
  `superseded_by: ["rec-b"]`. Nothing is dropped.
- `as_of` 2026-03-05: `rec-b` is excluded because it was not yet recorded; `rec-a` is `current`.

## Provenance
The two record texts are the `TARGET_A` / `TARGET_B` strings of the field-defect regression in
`tests/test_superseded_not_dropped.py`; the unrelated record is one of that file's filler
sentences. The `supersedes` and `valid_from` fields follow the request shape in
`docs/full-reference.md`. Record IDs, timestamps, and the `ops-notes` source are invented for
this example (real IDs are assigned by the service).

## Limits
- This shows the pure annotation logic, not the HTTP write path, the sqlite sidecar that
  persists supersession links, or ranking. The order of `relevance_order` is supplied by hand.
- `supersedes` is a caller declaration. Fidelis does not check that the correction is true, or
  that the old record was wrong.
- Near-identical old and new sentences can be collapsed by an earlier deduplication stage on
  the legacy `/recall` path (xfail test `test_recall_near_identical_pair_collapses_before_temporal_view`).
  This example uses dissimilar texts and does not exercise that gap.
- An empty or missing supersession index yields `"index": "unavailable"`, not "current".
