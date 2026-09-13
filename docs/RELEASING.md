# Releasing Fidelis Memory

## Version strategy

Fidelis uses Semantic Versioning for the installable `fidelis-memory` package.
While the major version is zero, the public API is still evolving:

- `0.1.x` is the supported 0.1 product line. Patch releases contain compatible
  fixes, documentation, packaging, and client-compatibility maintenance.
- `0.2.0` is reserved for a user-visible capability or boundary change that
  satisfies the outcome gates in [`ROADMAP.md`](../ROADMAP.md). Volume of work
  alone is not a reason to increment the minor version.
- Breaking pre-1.0 changes require a minor release, an upgrade note, and a
  recovery path. Patch releases must not knowingly break the 0.1 contract.
- `1.0.0` requires an explicitly stable API/configuration contract and a
  documented compatibility window; it is not implied by adoption or time.

The archived `cogito-ergo` entries in `CHANGELOG.md` describe a predecessor
package. They do not set the current Fidelis package version.

## Coordinated version surfaces

Every release must carry the same version in:

- `pyproject.toml` and `src/fidelis/__init__.py`;
- `server.json` and `gemini-extension.json`;
- `CITATION.cff` and `codemeta.json`;
- current install commands in `README.md`, `llms.txt`, and
  `docs/full-reference.md`; and
- the public-install assertions in `tests/test_public_install_truth.py`.

Historical changelog text and protocol-marker versions are not mechanically
rewritten.

## Release format

Use this order in the GitHub release body:

1. **Why this release:** one sentence naming the user outcome.
2. **Who it is for:** link to the user-fit matrix and state the supported OS and
   client boundary.
3. **Install or upgrade:** one pinned command plus the native Gemini route when
   relevant.
4. **What changed:** only shipped behavior, packaging, and documentation.
5. **Evidence:** CI/release checks and the release-readiness record.
6. **Known limits:** link to the README section; never hide a non-fit.
7. **Next:** link to the outcome-gated roadmap without presenting it as shipped.

For 0.1.0 use the title `Fidelis Memory 0.1.0 — local-first memory across five
agent clients` and the body in [`releases/0.1.0.md`](releases/0.1.0.md).

## Release sequence

1. Merge the coordinated version PR only after repository checks, Hermes Gate,
   and independent release rubric review pass.
2. Create annotated tag `vX.Y.Z` at the exact merged commit and push the tag.
3. Dispatch `.github/workflows/release.yml` with that existing tag. It verifies
   the tag/version match, lints, tests, builds, runs `twine check`, publishes to
   PyPI by OIDC, then publishes `server.json` to the MCP Registry.
4. Verify the GitHub Release, PyPI version, MCP Registry version, source archive,
   and Gemini extension manifest from public URLs. A green workflow without
   those read-backs is not a finished release.
5. Test one fresh wheel install and one client start from the public artifacts.
6. If publication fails after PyPI accepted the version, never overwrite it.
   Fix forward with a patch release and make the incomplete coordinate explicit
   in the release notes.
