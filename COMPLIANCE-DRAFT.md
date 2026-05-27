# COMPLIANCE-DRAFT.md — cogito-ergo

**Status:** DRAFT. Internal self-audit artifact generated 2026-04-17 by
`hermes_deliverable`. Not a conformity declaration; not a legal attestation.
Review and sign off before distribution.

**System:** cogito-ergo — semantic memory retrieval for AI agents. Retrieves
prior conversations via integer-pointer filter (LLM returns only integers,
structurally cannot corrupt returned content). Hybrid R@1 = 96.4% on
LongMemEval_S.

**Risk classification under EU AI Act:** Not a high-risk AI system per Annex III
on its own. Becomes high-risk when used as a retrieval layer inside a
high-risk downstream system (clinical decision, hiring, credit, law enforcement,
emergency triage). Compliance posture below assumes the downstream context may
be high-risk — compliance documentation written to meet the strictest case.

---

## Article 9 — Risk management & data provenance

**Data sources used by cogito-ergo:**

- Caller-provided conversation logs (not owned or stored by the library itself
  beyond the session the caller passes in).
- No training data. cogito-ergo is a retrieval library, not a trained model.
- Embedding model: `nomic-embed-text` via Ollama (local, open-weights).
- Filter LLM: caller-supplied. Default recommendation: small local model
  (qwen3.5:2b / 4b). See `README.md` for the full calibration.

**Risk identified:**
1. Retrieval may miss critical context (false negative). R@1 = 96.4%; 3.6%
   failure mode is a missed retrieval.
2. Retrieval may surface stale or contradicted content (staleness). Mitigation:
   caller must version memories or pass a freshness filter.
3. Embedding model drift if Ollama updates the nomic-embed-text pointer.
   Mitigation: pin model version in caller deployment.

**Residual risk:** acceptable for non-safety-critical uses. NOT acceptable as
sole source of truth for high-risk decisions; always combine with authoritative
retrieval.

## Article 10 — Data governance

- No PII stored by the library.
- Caller is responsible for the lawful basis of the memories they pass in.
- No bias metrics computed by cogito-ergo itself (retrieval is content-agnostic).
- Bias in retrieved content is a function of bias in the caller's memory store;
  cogito-ergo does not filter for bias.

## Article 14 — Human oversight & override

- **Override mechanism:** caller controls all inputs and outputs. cogito-ergo
  does not act; it returns ranked memory pointers. The caller's agent is where
  oversight is enforced.
- **Kill-switch:** library is a Python function call. Stopping it is `raise`.
- **Audit trail:** caller is responsible for logging retrieval events. See
  `examples/audit_logging.py` (TODO, not yet implemented — flagged as gap).

## Article 15 — Accuracy, robustness, cybersecurity

- **Accuracy:** benchmarked at R@1 = 96.4% on LongMemEval_S (runP-v35, 2026-04-18). Measurement
  methodology in `bench/`.
- **Robustness:** TODO — adversarial inputs (prompt injection in memory content)
  have not been systematically fuzzed. Flagged as gap. See `little-canary` tool
  for integration candidate.
- **Cybersecurity:** the integer-pointer filter is a structural mitigation —
  the filter LLM cannot hallucinate content; it can only pick or mis-pick an
  index. Worst case is mis-retrieval, not fabricated content.

## Article 86 — Right to explanation

For any retrieved memory, the caller receives:
- The integer index chosen by the filter LLM
- The similarity score of each candidate
- The full candidate list considered

This is enough for a downstream system to explain *why this memory and not
others*. cogito-ergo itself does not produce natural-language explanations.

## Known limitations (disclosure)

1. No bias testing (no test suite; flagged).
2. No behavioral robustness harness against memory-content injection (flagged).
3. No model registry; depends on caller pinning their embedding and filter
   model versions.
4. No incident-response runbook; failures are exceptions the caller must handle.
5. English-optimized; performance on non-English queries not measured.

## Remediation plan (next 90 days)

- Add adversarial memory-injection test suite (sprint 1)
- Add example audit-logging integration (sprint 1)
- Add calibration bench for non-English queries (sprint 2)
- Publish incident-response runbook for common failure modes (sprint 2)
