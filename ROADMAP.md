# Fidelis Memory roadmap

## 0.1 line: maintain the supported local contract

The 0.1 line receives compatible reliability, documentation, packaging, and
client-integration fixes. Its contract stays local-first, single-machine, and
pre-1.0.

## 0.2.0: outcome gates

0.2.0 is not a bucket for accumulated work. It is eligible only when the
release candidate makes at least one current non-fit materially usable and all
applicable gates below are evidenced.

### Outcome A — reach a verified first recall with less setup ambiguity

- A clean supported machine can progress from package install to a successful
  recall using one canonical path.
- `fidelis doctor` or an equivalent diagnostic identifies each missing local
  prerequisite and names the corrective action without exposing memory data.
- macOS and Ubuntu qualification runs are stored as reproducible receipts.

### Outcome B — make the data boundary explicit and recoverable

- The legacy `~/.cogito/` path has a documented migration design to a Fidelis
  namespace with backup, rollback, collision, and interrupted-run behavior.
- No automatic migration ships until tests prove old data remains readable and
  rollback restores the pre-migration state.
- `user_id` remains documented as a namespace unless a real authentication
  boundary is implemented and threat-modeled.

### Outcome C — convert one important non-fit into a supported fit

Choose and complete one, based on issue and adoption evidence:

- qualify a Windows service/install path;
- support an intentionally operated shared deployment with authentication and
  namespace isolation; or
- remove the local Ollama prerequisite through a clearly bounded alternative
  that preserves the privacy model it claims.

Selection is an evidence decision, not a promise that all three belong in
0.2.0.

### Outcome D — keep public truth synchronized

- Version and install coordinates are tested across package, registry,
  extension, citation, documentation, and release surfaces.
- Release claims identify their evidence class and do not present roadmap work
  as shipped.
- Public install, registry, archive, and one client-start read-back are required
  to close the release.

## Deliberately not assigned to 0.2.0

- a hosted Fidelis service;
- enterprise compliance certification;
- a stable 1.0 API promise; and
- arbitrary feature count or marketing deadlines.

Those require separate evidence and product decisions. Issues may explore them,
but their existence does not expand the 0.2.0 contract.
