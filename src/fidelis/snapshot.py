"""
cogito snapshot — compressed index layer (zer0dex-style).

Builds a structured markdown summary of the entire memory store — a semantic
table of contents that an agent loads once at session start. Solves the
cross-reference problem: the agent knows what categories exist and where
knowledge lives, without needing to query for it.

Architecture:
  MEMORY.md (in context, always)    — navigational scaffold, ~500-800 tokens
  Vector store (queried per-message) — fact retrieval, integer-pointer fidelity

The two layers are complementary:
  - Compressed index: cross-domain linking, cheap in tokens
  - Vector store: precise fact retrieval, exact wording preserved

Usage:
    cogito snapshot              # build and write snapshot
    cogito snapshot --dry-run    # preview without writing
    cogito snapshot --rebuild    # force rebuild even if snapshot exists
"""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


_SNAPSHOT_SYSTEM = (
    "You are building a compressed memory index for an AI agent. "
    "Read the numbered facts carefully. "
    "Produce a structured markdown file that groups the facts by project, topic, or category.\n\n"
    "Requirements:\n"
    "- ONLY use information from the provided facts. Do not invent or copy examples.\n"
    "- For each group: 1-3 bullet lines capturing key entities, names, tools, decisions, and vocabulary\n"
    "- Format: ## Heading then bullets with **Bold Label** — specific details\n"
    "- Prioritise: proper nouns, version numbers, incident names, dates, tool names\n"
    "- Skip generic filler ('the project is progressing well', 'research was done')\n"
    "- Aim for dense specific entries, 500-800 tokens total\n"
    "- Output ONLY the markdown. No preamble, no explanation, no code blocks.\n\n"
    "Common useful headings: Projects, Tools, Incidents, Architecture, People & Agents, "
    "Research, Business — use only what fits the actual facts."
)


def _sample_memories(memory: Any, user_id: str, n: int) -> list[str]:
    """Fetch all memories and return up to n randomly sampled texts."""
    try:
        raw = memory.get_all(user_id=user_id, limit=10000)  # type: ignore
        results = raw.get("results", [])
        texts = [r.get("memory", "") for r in results if r.get("memory")]
        if len(texts) > n:
            texts = random.sample(texts, n)
        return texts
    except Exception as e:
        raise RuntimeError(f"Failed to fetch memories: {e}") from e


def _build_snapshot(
    memories: list[str],
    endpoint: str,
    token: str,
    model: str,
    timeout: float,
) -> str:
    """Single LLM call to produce a structured markdown index from sampled memories."""
    lines = [f"[{i+1}] {m[:120].replace(chr(10), ' ')}" for i, m in enumerate(memories)]
    memories_block = "\n".join(lines)

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SNAPSHOT_SYSTEM},
            {"role": "user", "content": (
                f"Memory facts ({len(memories)} total):\n{memories_block}\n\n"
                "Produce the compressed markdown index from these facts:"
            )},
        ],
        "max_tokens": 2000,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )

    # Qwen3 thinking models need native Ollama API (think:false)
    is_ollama_local = "localhost:11434" in endpoint or "127.0.0.1:11434" in endpoint
    is_thinking_model = model.startswith("qwen3") or model.startswith("qwen3.5")

    if is_ollama_local and is_thinking_model:
        native_payload = json.dumps({
            "model": model,
            "messages": json.loads(payload.decode())["messages"],
            "think": False,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=native_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
            raw = result["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Native Ollama call failed: {e}") from e
    else:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
            raw = result["choices"][0]["message"]["content"].strip()
        except urllib.error.URLError as e:
            raise RuntimeError(f"LLM call failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}") from e

    # Strip thinking tokens
    if "<think>" in raw:
        end = raw.rfind("</think>")
        if end >= 0:
            after = raw[end + 8:].strip()
            raw = after if after else raw

    # Must start with a markdown heading
    if not raw.strip().startswith("#"):
        # Find first heading
        idx = raw.find("\n#")
        if idx >= 0:
            raw = raw[idx + 1:]

    return raw.strip()


def _snapshot_path(cfg: dict[str, Any]) -> Path:
    """Return path where snapshot.md should live, alongside the config file."""
    config_path_str = cfg.get("_config_file", "")
    if config_path_str:
        return Path(config_path_str).parent / "snapshot.md"
    return Path.home() / ".cogito" / "snapshot.md"


def _read_snapshot(cfg: dict[str, Any]) -> str | None:
    """Read existing snapshot, return None if it doesn't exist."""
    p = _snapshot_path(cfg)
    if p.exists():
        try:
            return p.read_text()
        except Exception:
            return None
    return None


def _write_snapshot(text: str, cfg: dict[str, Any]) -> Path:
    """Write snapshot to disk. Returns path written."""
    p = _snapshot_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def snapshot(
    memory: Any,
    cfg: dict[str, Any],
    n: int = 500,
    dry_run: bool = False,
    rebuild: bool = False,
) -> str:
    """
    Build compressed index. Returns the markdown text produced.

    Writes to snapshot.md unless dry_run=True.
    Skips rebuild if snapshot already exists unless rebuild=True.
    """
    from fidelis.recall import _resolve_filter_endpoint  # avoid circular at module level

    if not rebuild:
        existing = _read_snapshot(cfg)
        if existing:
            print(f"[cogito snapshot] Snapshot already exists at {_snapshot_path(cfg)}")
            print("[cogito snapshot] Use --rebuild to regenerate.")
            return existing

    endpoint, token = _resolve_filter_endpoint(cfg)
    if not endpoint:
        raise RuntimeError(
            "No filter endpoint configured. Set COGITO_FILTER_ENDPOINT + "
            "COGITO_FILTER_TOKEN, or ANTHROPIC_API_KEY."
        )

    model = cfg.get("calibrate_model", cfg.get("filter_model", "mistral:7b"))
    timeout = max(cfg.get("filter_timeout_ms", 30000), 120000) / 1000
    user_id = cfg.get("user_id", "agent")

    print(f"[cogito snapshot] Sampling memories (n={n})...")
    memories = _sample_memories(memory, user_id, n)
    if not memories:
        raise RuntimeError("No memories in store. Run `cogito seed` first.")
    print(f"[cogito snapshot] Sampled {len(memories)} memories. Calling {model}...")

    text = _build_snapshot(memories, endpoint, token, model, timeout)
    lines = text.count("\n") + 1
    tokens_est = len(text.split())
    print(f"[cogito snapshot] Generated {lines} lines (~{tokens_est} tokens).")

    if dry_run:
        print("[cogito snapshot] DRY RUN — not writing.\n")
        print(text)
        return text

    path = _write_snapshot(text, cfg)
    print(f"[cogito snapshot] Written to {path}")
    return text
