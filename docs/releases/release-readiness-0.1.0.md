# Fidelis Memory 0.1.0 release readiness

Status: **producer PASS — independent review required before tagging**

This is a binary rubric. Every row must be `PASS`; a `PARTIAL`, `UNKNOWN`, or
`FAIL` blocks tagging. The producer records evidence, then an independent
reviewer tests the same candidate diff. Publication read-backs are evaluated
after tagging and remain necessary for release completion.

| Gate | Pass condition | Candidate evidence | Status |
|---|---|---|---|
| Version coherence | Every maintained current-version surface resolves to `0.1.0`; historical versions stay historical | Focused version/metadata tests and exact repository scan | PASS |
| Package build | Wheel and sdist build; `twine check` accepts both | `fidelis_memory-0.1.0` wheel and sdist built; both passed `twine check` | PASS |
| Fresh install | A clean environment installs the built wheel and imports the `0.1.0` package | Isolated Python 3.12 wheel install, import/version assertion, and CLI-help smoke | PASS |
| Core behavior | The release test suite and lint gate pass | Ruff passed; 553 tests passed and 3 skipped under the release exclusions; the live local-Ollama SIGTERM case was deselected locally and remains in hosted CI | PASS |
| Client contract | Codex/Claude, Copilot, Gemini, and OpenClaw install or manifest contracts remain covered | 17 focused release/version/client tests passed; broader suite passed | PASS |
| User usability | A reader can decide whether to install, follow one supported path, and find verification/troubleshooting | `README.md`; `docs/user-fit.md` | PASS |
| Claim discipline | Numerical, privacy, compatibility, and limitation language is scoped to checked-in evidence | Claims narrowed to local observations; unsupported competitor, cost, privacy, time, and latency generalizations removed | PASS |
| Release operation | Tag/version mismatch fails closed; PyPI and MCP publication run only after tests/build | `.github/workflows/release.yml`; `docs/RELEASING.md` | PASS |
| Recovery | A failed publication cannot overwrite PyPI and has an explicit fix-forward rule | `docs/RELEASING.md` | PASS |
| Scope separation | 0.1.0 describes shipped behavior; 0.2.0 work is outcome-gated and labeled prospective | `CHANGELOG.md`; `ROADMAP.md` | PASS |

Producer observations above are bound to the candidate diff reviewed in the
release pull request. Hosted matrix results and public artifact read-backs are
separate completion evidence; neither is inferred from these local checks.

## Independent decision

The reviewer returns exactly one release decision:

- `PASS`: all ten gates are evidenced and no release-blocking finding remains;
- `REVISE`: one or more named gates can pass after a bounded candidate change;
  or
- `BLOCK`: the version should not be tagged without a material product or
  architecture decision.

The producer does not self-license this release.
