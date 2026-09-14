# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in fidelis (the `fidelis-memory`
package), please report it responsibly.

**Do not open a public issue for security vulnerabilities.**

Instead, email us at: **roli@hermes-labs.ai**

Include:
- A description of the vulnerability.
- Steps to reproduce the issue.
- Any relevant logs or output.

## Scope

fidelis is a local-first memory and retrieval service for AI agents (Claude
Code, Codex, GitHub Copilot CLI, Gemini CLI, OpenClaw). It stores notes in a
local Chroma + SQLite store and serves retrieval through a local HTTP service
and an MCP server; the default retrieval path makes no outbound model API
call. Reports about the local service, the MCP server, the CLI, the
per-client installation writers (Codex, Claude Code, Copilot CLI, Gemini CLI,
OpenClaw config handling), and the published PyPI package are in scope.
fidelis does not operate a hosted service, so reports about infrastructure it
does not run are out of scope.

## Response Timeline

- **Acknowledgment**: Within 48 hours of your report.
- **Assessment**: Within 7 days we will confirm the issue and outline next
  steps.
- **Fix**: We aim to release a patch within 30 days of confirmation.

## Supported Versions

Security updates are applied to the latest release only. fidelis is
pre-release software (0.1.0); expect breaking changes between versions.

Thank you for helping keep fidelis safe.
