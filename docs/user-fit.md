# Is Fidelis Memory a fit?

Fidelis 0.1.0 has one job: let an agent retrieve the original passages from
notes you keep on your own machine. The default retrieval path does not call an
LLM. Your agent still uses its normal model to answer from the retrieved text.

## Fit matrix

| Your situation | 0.1.0 fit | What you get | Boundary to know |
|---|---|---|---|
| Solo developer with Markdown notes on macOS or Ubuntu | Yes | Local ingestion, retrieval, and service startup | Ollama and the local embedding model are prerequisites |
| Codex or Claude Code user who wants memory across sessions | Yes | A supported MCP installation path and recall/orientation tools | The client decides when to call the tools |
| GitHub Copilot CLI, Gemini CLI, or OpenClaw user | Yes, with the documented client version and setup | A client-owned or native extension registration path with read-back checks | First launch can need dependency-cache warm-up; see the client section in the README |
| Application author who can call a local HTTP service or Python helper | Yes, for a single-machine deployment | Verbatim passages through the documented local interfaces | APIs remain pre-1.0 and should be version-pinned |
| User who cannot run Ollama or download the embedding model | No | — | 0.1.0 does not provide a hosted embedding or memory service |
| Windows-only user | Not yet a supported fit | The Python package may run, but this release does not claim a gate-tested service install | Use a macOS or Ubuntu environment, or help qualify the Windows path |
| Team needing a shared, centralized memory service | Not yet | — | 0.1.0 is local and single-machine; centralized operation is not a supported contract |
| Team needing per-tenant authorization or managed multi-user isolation | No | — | The `user_id` value is a namespace, not an identity or authorization boundary |
| Regulated deployment seeking a compliance guarantee | Evaluation required | Local-first defaults can reduce third-party exposure | Fidelis is not a compliance certification; the deployer owns its assessment |
| Workflow that needs generated summaries instead of source passages | Usually no | Fidelis intentionally returns stored passages without rewriting them | Let the calling agent summarize after retrieval, or use a different memory product |

## Choose Fidelis when

- the source wording matters;
- your working memory is already in local notes or sessions;
- you prefer a local service and can operate its prerequisites; and
- your agent host supports MCP, or your application can use the local API.

Choose another approach when the primary need is a hosted team service,
cross-tenant authorization, managed compliance, or automatic synthesis of a
shared knowledge base.

## The supported 0.1.0 path

1. Install the pinned `fidelis-memory` distribution.
2. Run `fidelis init` and add local material with `fidelis watch`.
3. Register Fidelis with one documented agent client, or run the MCP server.
4. Verify the service with `fidelis health` and ask the client to recall a
   distinctive passage you stored.

If that path fails on a supported environment, open an issue with the OS,
Python version, client and client version, command used, and the redacted error
output. Do not include memory contents or credentials.
