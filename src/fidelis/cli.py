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
    # Delegate to server.main()
    from fidelis.server import main as server_main
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
    p_server = sub.add_parser("server", help="Start the cogito server")
    p_server.set_defaults(func=cmd_server)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
