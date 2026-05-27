# Roadmap: code-grounded read of the retrieval fork

## 1. Is the per-type-index design sound?

Partly. The core idea, separate storage for atomic facts vs session-like memories, is sound because the current code already has two materially different retrieval assumptions: `/recall` is built around short candidate snippets and integer-pointer filtering ([src/cogito/recall.py:67-105](src/cogito/recall.py:67)), while `/recall_hybrid` adds BM25, prefixed embeddings, and route-specific reranking ([src/cogito/recall_hybrid.py:203-317](src/cogito/recall_hybrid.py:203), [src/cogito/recall_hybrid.py:519-637](src/cogito/recall_hybrid.py:519)).

What is missing in the design and would bite:

- Insert-time routing is not wired today. `/store` writes directly to one Chroma collection with payload `{"data": text, "user_id": user_id}` and no `mem_type` metadata ([src/cogito/server.py:195-209](src/cogito/server.py:195)). `mem0_config()` only builds one collection name (`cfg["collection"]`) ([src/cogito/config.py:142-167](src/cogito/config.py:142)). `/add` delegates straight to `memory.add()` with no type hook ([src/cogito/server.py:213-223](src/cogito/server.py:213)).
- The proposed default “query both, merge by score” is underspecified. Both paths expose cosine-like scores after stage 1 (`recall_b` rewrites `score` to cosine at [src/cogito/recall_b.py:280-289](src/cogito/recall_b.py:280); hybrid does the same at [src/cogito/recall_hybrid.py:299-316](src/cogito/recall_hybrid.py:299)), but LLM rerank changes order without recomputing score in either path ([src/cogito/recall.py:61-64](src/cogito/recall.py:61), [src/cogito/recall.py:220-227](src/cogito/recall.py:220), [src/cogito/recall_hybrid.py:419-425](src/cogito/recall_hybrid.py:419), [src/cogito/recall_hybrid.py:507-513](src/cogito/recall_hybrid.py:507)). “Merge by score” would therefore be incorrect after reranking.
- Migration is nontrivial because `seed.py` currently assumes atomized fact writes to `/store` ([src/cogito/seed.py:328-349](src/cogito/seed.py:328)); the design does not define how existing curated facts, `/add`-extracted facts, and future long-form writes are distinguished or backfilled.
- The 400-char split is explicitly unmeasured in the design, and the repo contains no production length-distribution measurement yet ([docs/DISPATCHER_DESIGN.md:56-57](docs/DISPATCHER_DESIGN.md:56)).

## 2. Is the skeptic right about the 21pt regression?

On the regression itself: yes. `bench/eval.py` runs `/recall_hybrid` directly as mode `E` ([bench/eval.py:358-368](bench/eval.py:358)), and the status/docs consistently report 54% vs 75%.

On the architect’s diagnosis: only partly proven from code.

- “BM25 hurts short facts” is plausible and code-consistent. The hybrid stage always fuses BM25 runs into RRF when `bm25s` is present ([src/cogito/recall_hybrid.py:264-303](src/cogito/recall_hybrid.py:264)). That is structurally different from `/recall`, which uses multi-query dense recall plus cosine rerank and no BM25 ([src/cogito/recall.py:48-64](src/cogito/recall.py:48), [src/cogito/recall_b.py:313-343](src/cogito/recall_b.py:313)). On a corpus of short fact strings, extra lexical runs can easily perturb rank-1.
- “150-char filter truncation hurts session content” is true as a code fact: `/recall` shows only 150 chars per candidate to the filter ([src/cogito/recall.py:87-91](src/cogito/recall.py:87)). But it does **not** explain the 21pt regression on cogito’s atomic eval, because that eval compares `/recall` against `/recall_hybrid`, and hybrid mostly keeps Stage 1 order for default-route queries at `tier="filter"` ([src/cogito/recall_hybrid.py:607-613](src/cogito/recall_hybrid.py:607)). The 21pt drop is therefore much more likely a Stage 1 mismatch than a filter-snippet issue.
- For session retrieval, the bigger code-grounded mismatch is elsewhere: the 96.4% benchmark (runP-v35) relies on turn chunking and dedup back to sessions ([bench/longmemeval_combined_pipeline_flagship.py:140-188](bench/longmemeval_combined_pipeline_flagship.py:140)), while the production port explicitly omits turn chunking and session-date scaffolds ([src/cogito/recall_hybrid.py:20-24](src/cogito/recall_hybrid.py:20)). That gap is more important than the 150-char fact.

## 3. Fork recommendation

Fast-ship is the right fork.

Reason: the codebase today cleanly supports the narrow claim that `/recall_hybrid` is an opt-in LongMemEval-derived path and leaves `/recall` untouched ([src/cogito/server.py:175-193](src/cogito/server.py:175), [CHANGELOG.md:3-15](CHANGELOG.md:3)). It does **not** yet support the broader dispatcher claim because typed writes, migration, dual-index querying, and valid cross-path merge semantics are missing. Shipping the benchmark result as workload-specific is honest; trying to ship workload-aware dispatch now would be architecture-first, evidence-second.

## 4. Concrete engineering roadmap

1. Measure real memory length/type distribution before tuning thresholds.
Where: add a bench/CLI script over `memory.get_all()` like `snapshot.py`/`calibrate.py` do ([src/cogito/snapshot.py:53](src/cogito/snapshot.py:53), [src/cogito/calibrate.py:53](src/cogito/calibrate.py:53)).
Acceptance: report count, p50/p90/p99 length, newline distribution, turn-marker rate, and provisional buckets `<200`, `200-399`, `400-799`, `800+`.
Effort: 2-3h.

2. Add typed-write plumbing without changing default behavior.
Where: `src/cogito/server.py`, `src/cogito/config.py`, new classifier helper.
Acceptance: `/store` returns `mem_type`; config can target legacy single collection or typed collections; existing installs still default to legacy mode.
Effort: 5-7h.

3. Implement explicit migration, not implicit dispatch.
Where: new CLI command plus reuse of `memory.get_all()`/write paths.
Acceptance: migrate legacy collection to `{collection}_atomic` and `{collection}_session`, verify count parity, leave legacy untouched until success.
Effort: 6-8h.

4. Fix cross-path merge semantics before any dual-index default.
Where: dispatcher layer around `recall()` and `recall_hybrid()`.
Acceptance: merged results use source-aware ordering, not stale cosine scores after rerank; add tests proving LLM-reranked items are not re-sorted by obsolete `score`.
Effort: 4-6h.

5. Add medium-length and structured-memory falsifier tests.
Where: `bench/eval.py` plus new fixture file alongside `bench/eval_cases.json`.
Acceptance: dedicated cases for 200-800 char narratives and JSON/YAML/code blobs; report Path A vs Path B separately.
Effort: 4-6h.

6. Add paraphrase and typo perturbation tests.
Where: bench harness only; no benchmark rerun required for roadmap, but this is the next measurement gate.
Acceptance: reproducible generated variants and per-path pass/fail thresholds checked in.
Effort: 4-5h.

7. Only then prototype dispatcher policy.
Where: query-time dispatch layer plus API surface.
Acceptance: measured improvement over legacy on typed corpora, no regression below current `/recall` floor on atomic eval, and no claim of “auto” until tests above pass.
Effort: 6-10h.

## 5. What to commit now vs hold

Commit now: the opt-in `v0.3.0` hybrid path as an explicitly workload-specific feature. The server endpoint is isolated, `/recall` is unchanged, and the changelog already describes it that way ([CHANGELOG.md:5-15](CHANGELOG.md:5)).

Hold: any dispatcher, “generalizable” positioning, or default-routing changes. The current repo does not yet implement the storage split or the merge semantics needed to support that claim.

## 6. What the agents missed

- The benchmark/production mismatch is larger than “150-char truncation.” The benchmark’s win depends on turn chunking and session-level dedup ([bench/longmemeval_combined_pipeline_flagship.py:140-188](bench/longmemeval_combined_pipeline_flagship.py:140)); production hybrid explicitly removed those pieces ([src/cogito/recall_hybrid.py:20-24](src/cogito/recall_hybrid.py:20)).
- The dispatcher design assumes score comparability that the code does not preserve after reranking.
- `bench/results-guard-2026-04-17.json` is only 30 questions, not a full verification run, so it should not be used as evidence of the final claim.
- `bench/eval.py` scores positive cases by keyword hit threshold, not exact answer matching ([bench/eval.py:24-33](bench/eval.py:24), [bench/eval.py:180-201](bench/eval.py:180)). That is fine for regression tracking, but it widens uncertainty when arguing about small deltas or cross-workload transfer.
