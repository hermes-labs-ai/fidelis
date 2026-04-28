"""
fidelis CLI

  fidelis init                       install + start service (launchd/systemd)
  fidelis watch  ~/notes             auto-ingest a directory
  fidelis mcp install                wire Claude Code MCP integration
  fidelis recall "query"             two-stage recall via running server
  fidelis query  "query"             simple vector query (no filter)
  fidelis add    "text"              add a memory
  fidelis seed   ~/memory/ ~/notes/  bulk-seed from markdown files
  fidelis health                     check server health
  fidelis server                     start the server (alias for fidelis-server)

All commands talk to the HTTP server. After `fidelis init` the service runs
under your OS service manager (launchd on macOS, systemd on Linux).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import os


def _base_url() -> str:
    # FIDELIS_PORT preferred; COGITO_PORT kept as backwards-compat alias.
    port = os.environ.get("FIDELIS_PORT") or os.environ.get("COGITO_PORT", "19420")
    return f"http://127.0.0.1:{port}"


def _server_error(exc: BaseException) -> None:
    """Print a clean error + exit. Catches the full family of socket-level
    failures so users never see a Python traceback for server-side issues."""
    msg = str(exc) or type(exc).__name__
    print(f"Error: fidelis-server unreachable or unhealthy at {_base_url()}", file=sys.stderr)
    print(f"  reason: {msg}", file=sys.stderr)
    print("  • If you haven't installed the service: `fidelis init`", file=sys.stderr)
    print("  • If the service is installed: `tail ~/.fidelis/server.log`", file=sys.stderr)
    sys.exit(1)


def _post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{_base_url()}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        _server_error(e)


def _get(path: str) -> dict:
    try:
        with urllib.request.urlopen(f"{_base_url()}{path}", timeout=5) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        _server_error(e)


def _print_memories(memories: list, method: str = ""):
    if not memories:
        print("No memories found." + (f" (method: {method})" if method else ""))
        return
    tag = f" [{method}]" if method else ""
    print(f"{len(memories)} memories{tag}:\n")
    for i, m in enumerate(memories, 1):
        score = f"  score {m['score']:.3f}" if "score" in m else ""
        print(f"  [{i}]{score}")
        print(f"      {m['text']}")
        print()


def cmd_recall(args):
    payload = {
        "text": args.query,
        "limit": args.limit,
        "threshold": args.threshold,
    }
    if args.since:
        payload["since"] = args.since
    result = _post("/recall", payload)
    if args.raw:
        print(json.dumps(result, indent=2))
        return
    _print_memories(result.get("memories", []), result.get("method", ""))


def cmd_recall_hybrid(args):
    payload = {
        "text": args.query,
        "limit": args.limit,
        "tier": args.tier,
        "top_k": args.top_k,
    }
    result = _post("/recall_hybrid", payload)
    if args.raw:
        print(json.dumps(result, indent=2))
        return
    _print_memories(result.get("memories", []), result.get("method", ""))


def cmd_query(args):
    result = _post("/query", {"text": args.query, "limit": args.limit})
    if args.raw:
        print(json.dumps(result, indent=2))
        return
    _print_memories(result.get("memories", []))


def cmd_add(args):
    text = " ".join(args.text)
    result = _post("/add", {"text": text})
    print(f"Added {result.get('count', 0)} memories.")
    for m in result.get("memories", []):
        print(f"  → {m}")


def cmd_health(args):
    result = _get("/health")
    status = result.get("status", "unknown")
    count = result.get("count", "?")
    version = result.get("version", "?")
    calibrated = "yes" if result.get("calibrated") else "no"
    has_snapshot = "yes" if result.get("snapshot") else "no"
    print(f"status: {status}  |  memories: {count}  |  version: {version}  |  calibrated: {calibrated}  |  snapshot: {has_snapshot}")


def cmd_seed(args):
    from pathlib import Path
    from fidelis.seed import seed

    from fidelis.config import load
    sources = [Path(s) for s in args.sources]
    cfg = load()
    seed(
        sources=sources,
        base_url=_base_url(),
        cfg=cfg,
        glob_pattern=args.glob,
        dry_run=args.dry_run,
        force=args.force,
        verbose=args.verbose,
        delay_ms=args.delay,
        use_add=args.add,
    )


def cmd_snapshot(args):
    import os
    import sys
    from fidelis.config import load, mem0_config
    from fidelis.snapshot import snapshot

    cfg = load(args.config)

    site = os.environ.get("COGITO_SITE_PACKAGES")
    if site and site not in sys.path:
        sys.path.insert(0, site)

    from mem0 import Memory  # type: ignore
    memory = Memory.from_config(mem0_config(cfg))

    snapshot(memory, cfg, n=args.sample, dry_run=args.dry_run, rebuild=args.rebuild)


def cmd_calibrate(args):
    import os
    import sys
    from fidelis.config import load, mem0_config
    from fidelis.calibrate import calibrate

    cfg = load(args.config)

    site = os.environ.get("COGITO_SITE_PACKAGES")
    if site and site not in sys.path:
        sys.path.insert(0, site)

    from mem0 import Memory  # type: ignore
    memory = Memory.from_config(mem0_config(cfg))

    calibrate(memory, cfg, n=args.sample, dry_run=args.dry_run)


def cmd_server(args):
    # Delegate to server.main(). Strip the consumed "server" subcommand
    # (and any remaining argv) so the server's own argparse doesn't see it.
    import sys as _sys
    _sys.argv = [_sys.argv[0]]
    from fidelis.server import main as server_main
    server_main()


def main():
    parser = argparse.ArgumentParser(
        prog="fidelis",
        description="fidelis — agent memory with zero-LLM retrieval and a $0-incremental QA scaffold",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # recall
    p_recall = sub.add_parser("recall", help="Two-stage recall with LLM filter")
    p_recall.add_argument("query")
    p_recall.add_argument("--limit", type=int, default=50)
    p_recall.add_argument("--threshold", type=float, default=400.0)
    p_recall.add_argument("--since", help="ISO 8601 date string to filter memories created after this date (e.g., 2026-04-01)")
    p_recall.add_argument("--raw", action="store_true")
    p_recall.set_defaults(func=cmd_recall)

    # recall-hybrid (BM25 + dense + RRF + tiered LLM)
    p_hybrid = sub.add_parser(
        "recall-hybrid",
        help="Hybrid BM25+dense+RRF recall. Zero-LLM default (83.2%% R@1); opt-in LLM tiers for benchmark replication.",
    )
    p_hybrid.add_argument("query")
    p_hybrid.add_argument("--limit", type=int, default=50)
    p_hybrid.add_argument(
        "--tier", choices=["zero_llm", "filter", "flagship"], default="zero_llm",
        help="Retrieval tier: zero_llm (default, 83.2%% R@1, $0) | filter (benchmark-tuned) | flagship",
    )
    p_hybrid.add_argument("--top-k", type=int, default=5, help="Candidates shown to reranker")
    p_hybrid.add_argument("--raw", action="store_true")
    p_hybrid.set_defaults(func=cmd_recall_hybrid)

    # query
    p_query = sub.add_parser("query", help="Simple vector query (no filter)")
    p_query.add_argument("query")
    p_query.add_argument("--limit", type=int, default=5)
    p_query.add_argument("--raw", action="store_true")
    p_query.set_defaults(func=cmd_query)

    # add
    p_add = sub.add_parser("add", help="Add a memory")
    p_add.add_argument("text", nargs="+")
    p_add.set_defaults(func=cmd_add)

    # health
    p_health = sub.add_parser("health", help="Check server health")
    p_health.set_defaults(func=cmd_health)

    # seed
    p_seed = sub.add_parser("seed", help="Bulk-seed store from markdown/text files")
    p_seed.add_argument("sources", nargs="+", help="Dirs or files to seed from")
    p_seed.add_argument("--glob", default="*.md", help="File pattern (default: *.md)")
    p_seed.add_argument("--dry-run", action="store_true", help="Show what would be sent, don't write")
    p_seed.add_argument("--force", action="store_true", help="Re-seed even unchanged files")
    p_seed.add_argument("--verbose", "-v", action="store_true")
    p_seed.add_argument("--delay", type=int, default=0, help="ms between /store calls (default: 0)")
    p_seed.add_argument("--add", action="store_true", help="Use /add (mem0 extraction) instead of agent-curated /store")
    p_seed.set_defaults(func=cmd_seed)

    # snapshot
    p_snap = sub.add_parser("snapshot", help="Build compressed index (zer0dex-style MEMORY.md layer)")
    p_snap.add_argument("--sample", type=int, default=500, help="Memories to sample (default: 500)")
    p_snap.add_argument("--dry-run", action="store_true", help="Preview without writing")
    p_snap.add_argument("--rebuild", action="store_true", help="Force rebuild even if snapshot exists")
    p_snap.add_argument("--config", help="Path to .cogito.json")
    p_snap.set_defaults(func=cmd_snapshot)

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Extract vocab bridge from corpus (one-time)")
    p_cal.add_argument("--sample", type=int, default=200, help="Number of memories to sample (default: 200)")
    p_cal.add_argument("--dry-run", action="store_true", help="Preview mappings, don't write config")
    p_cal.add_argument("--config", help="Path to .cogito.json")
    p_cal.set_defaults(func=cmd_calibrate)

    # server
    p_server = sub.add_parser("server", help="Start the fidelis server")
    p_server.set_defaults(func=cmd_server)

    # init — install + start fidelis-server as a system service
    p_init = sub.add_parser(
        "init",
        help="Install + start fidelis-server as a launchd/systemd service (auto-starts on reboot)",
    )
    p_init.add_argument("--uninstall", action="store_true",
                        help="Stop service + remove the unit/plist")
    p_init.set_defaults(func=lambda a: sys.exit(_cmd_init(a)))

    # watch — auto-ingest a directory
    p_watch = sub.add_parser("watch", help="Auto-ingest markdown/text files from a directory")
    p_watch.add_argument("path", help="Directory to watch")
    p_watch.add_argument("--glob", nargs="+", default=None,
                         help="Glob patterns (default: *.md *.txt)")
    p_watch.add_argument("--max-files", type=int, default=500,
                         help="Initial-scan cap (default: 500)")
    p_watch.add_argument("--interval", type=float, default=5.0,
                         help="Poll interval in seconds (default: 5.0)")
    p_watch.add_argument("--once", action="store_true",
                         help="Initial scan only, don't poll continuously")
    p_watch.add_argument("--verbose", "-v", action="store_true")
    p_watch.set_defaults(func=lambda a: sys.exit(_cmd_watch(a)))

    # mcp — manage Claude Code MCP integration
    p_mcp = sub.add_parser("mcp", help="Manage Claude Code MCP integration")
    mcp_sub = p_mcp.add_subparsers(dest="mcp_command", required=True)
    p_mcp_install = mcp_sub.add_parser("install", help="Install fidelis MCP server into ~/.claude/settings.local.json")
    p_mcp_install.add_argument("--settings", help="Path to settings.local.json (default: ~/.claude/settings.local.json)")
    p_mcp_install.add_argument("--force", action="store_true",
                               help="Overwrite an existing 'fidelis' entry even if it doesn't look like ours")
    p_mcp_install.set_defaults(func=lambda a: sys.exit(_cmd_mcp_install(a)))
    p_mcp_uninstall = mcp_sub.add_parser("uninstall", help="Remove fidelis MCP server from settings")
    p_mcp_uninstall.add_argument("--settings", help="Path to settings.local.json")
    p_mcp_uninstall.set_defaults(func=lambda a: sys.exit(_cmd_mcp_uninstall(a)))

    args = parser.parse_args()
    args.func(args)


# Lazy-imports for the new consumer-surface commands so the existing CLI startup
# isn't slowed by importing platform-specific modules.

def _cmd_init(args):
    from fidelis.init_cmd import cmd_init
    return cmd_init(args)


def _cmd_watch(args):
    from fidelis.watch_cmd import cmd_watch
    return cmd_watch(args)


def _cmd_mcp_install(args):
    from fidelis.mcp_cmd import cmd_mcp_install
    return cmd_mcp_install(args)


def _cmd_mcp_uninstall(args):
    from fidelis.mcp_cmd import cmd_mcp_uninstall
    return cmd_mcp_uninstall(args)


if __name__ == "__main__":
    main()
