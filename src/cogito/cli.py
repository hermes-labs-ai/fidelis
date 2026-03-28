"""
cogito CLI

  cogito recall "query"              two-stage recall via running server
  cogito query  "query"              simple vector query (no filter)
  cogito add    "text"               add a memory
  cogito seed   ~/memory/ ~/notes/   bulk-seed from markdown files
  cogito health                      check server health
  cogito server                      start the server (alias for cogito-server)

All commands talk to the HTTP server. Server must be running separately
(cogito-server) or via your process manager of choice.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import os


def _base_url() -> str:
    port = os.environ.get("COGITO_PORT", "19420")
    return f"http://127.0.0.1:{port}"


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
    except urllib.error.URLError:
        print(f"Error: cogito server not reachable at {_base_url()}", file=sys.stderr)
        print("Start it with: cogito-server", file=sys.stderr)
        sys.exit(1)


def _get(path: str) -> dict:
    try:
        with urllib.request.urlopen(f"{_base_url()}{path}", timeout=5) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError:
        print(f"Error: cogito server not reachable at {_base_url()}", file=sys.stderr)
        sys.exit(1)


def _print_memories(memories: list, method: str = ""):
    if not memories:
        print("No memories found." + (f" (method: {method})" if method else ""))
        return
    tag = f" [{method}]" if method else ""
    print(f"{len(memories)} memories{tag}:\n")
    for i, m in enumerate(memories, 1):
        score = f"  score {m['score']:.0f}" if "score" in m else ""
        print(f"  [{i}]{score}")
        print(f"      {m['text']}")
        print()


def cmd_recall(args):
    result = _post("/recall", {
        "text": args.query,
        "limit": args.limit,
        "threshold": args.threshold,
    })
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
    print(f"status: {status}  |  memories: {count}  |  version: {version}")


def cmd_seed(args):
    from pathlib import Path
    from cogito.seed import seed

    from cogito.config import load
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


def cmd_calibrate(args):
    import os
    import sys
    from cogito.config import load, mem0_config
    from cogito.calibrate import calibrate

    cfg = load(args.config)

    site = os.environ.get("COGITO_SITE_PACKAGES")
    if site and site not in sys.path:
        sys.path.insert(0, site)

    from mem0 import Memory  # type: ignore
    memory = Memory.from_config(mem0_config(cfg))

    calibrate(memory, cfg, n=args.sample, dry_run=args.dry_run)


def cmd_server(args):
    # Delegate to server.main()
    from cogito.server import main as server_main
    server_main()


def main():
    parser = argparse.ArgumentParser(
        prog="cogito",
        description="cogito-ergo — memory layer for AI agents",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # recall
    p_recall = sub.add_parser("recall", help="Two-stage recall with LLM filter")
    p_recall.add_argument("query")
    p_recall.add_argument("--limit", type=int, default=50)
    p_recall.add_argument("--threshold", type=float, default=400.0)
    p_recall.add_argument("--raw", action="store_true")
    p_recall.set_defaults(func=cmd_recall)

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

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Extract vocab bridge from corpus (one-time)")
    p_cal.add_argument("--sample", type=int, default=200, help="Number of memories to sample (default: 200)")
    p_cal.add_argument("--dry-run", action="store_true", help="Preview mappings, don't write config")
    p_cal.add_argument("--config", help="Path to .cogito.json")
    p_cal.set_defaults(func=cmd_calibrate)

    # server
    p_server = sub.add_parser("server", help="Start the cogito server")
    p_server.set_defaults(func=cmd_server)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
