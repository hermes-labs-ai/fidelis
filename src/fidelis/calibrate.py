"""
cogito calibrate — one-time vocabulary bridge extraction.

Samples memories from the store, asks the filter LLM to identify vocabulary
gaps between natural language queries and stored technical facts, and writes
a vocab_map to .cogito.json.

At query time, recall_b uses the vocab_map for zero-LLM expansion:
  "freeze" → ["timeout", "cascade", "ollama"]
  "adoption" → ["downloads", "PyPI", "installs"]

Run once after initial seeding, and optionally after large corpus updates.

Usage:
    cogito calibrate
    cogito calibrate --sample 300 --dry-run
"""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


_CALIBRATE_SYSTEM = (
    "You are building a vocabulary bridge for a technical memory retrieval system. "
    "Users ask questions in plain English. Stored facts use technical jargon, "
    "proper nouns, acronyms, and domain-specific terms.\n\n"
    "Your job: identify pairs where a plain-English query word would NOT appear "
    "in the stored fact, but the stored fact IS the right answer.\n\n"
    "Output ONLY a valid JSON object. "
    "Keys: plain-English words a user might type in a query. "
    "Values: arrays of SHORT TECHNICAL TERMS (1-5 words each, NEVER full sentences) "
    "that appear verbatim in the stored facts and carry the same meaning as the key. "
    "Do NOT include full sentences as values. "
    "Do NOT include generic stop words. "
    "Max 60 pairs. Focus on genuine vocabulary gap cases only.\n\n"
    'Example: {"freeze": ["timeout", "cascade", "blocked"], '
    '"adoption": ["downloads", "PyPI", "installs"], '
    '"broken": ["flat scores", "metadata missing", "zero results"], '
    '"memory not found": ["flat score", "user_id", "chromadb"], '
    '"agent stuck": ["timeout", "seed", "ollama cascade"]}'
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


def _build_vocab_map(
    memories: list[str],
    endpoint: str,
    token: str,
    model: str,
    timeout: float,
) -> dict[str, list[str]]:
    """Single LLM call to extract vocabulary bridge from sampled memories."""
    lines = [f"[{i+1}] {m[:120].replace(chr(10), ' ')}" for i, m in enumerate(memories)]
    memories_block = "\n".join(lines)

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _CALIBRATE_SYSTEM},
            {"role": "user", "content": f"Memory facts:\n{memories_block}\n\nOutput the vocabulary bridge JSON:"},
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

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        raw = result["choices"][0]["message"]["content"].strip()
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM call failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}") from e

    # Strip thinking tokens (<think>...</think>)
    # Some models (qwen3, deepseek-r1) put reasoning in <think> and answer after.
    # If nothing comes after </think>, fall back to looking inside the block.
    if "<think>" in raw:
        end = raw.rfind("</think>")
        if end >= 0:
            after = raw[end + 8:].strip()
            raw = after if after else raw  # keep full raw if nothing after </think>


    # Extract JSON object
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        raise RuntimeError(f"No JSON object in LLM output: {raw[:200]}")

    try:
        vocab_map = json.loads(raw[start:end])
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON parse failed: {e}\nRaw: {raw[:300]}") from e

    if not isinstance(vocab_map, dict):
        raise RuntimeError(f"Expected dict, got {type(vocab_map)}")

    # Normalise: lowercase keys, flatten values to list[str]
    clean: dict[str, list[str]] = {}
    for k, v in vocab_map.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, list):
            clean[k.lower().strip()] = [str(t) for t in v if t]
        elif isinstance(v, str) and v:
            clean[k.lower().strip()] = [v]

    return clean


def _write_vocab_map(vocab_map: dict[str, list[str]], cfg: dict[str, Any]) -> Path:
    """Merge vocab_map into the config file. Returns the path written."""
    config_path_str = cfg.get("_config_file", "")

    if config_path_str:
        config_path = Path(config_path_str)
    else:
        # No file loaded — write to default location
        config_path = Path.home() / ".cogito" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing config (preserve all other keys)
    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except Exception:  # noqa: silent — corrupted config → write fresh on save
            pass

    existing["vocab_map"] = vocab_map

    # Remove internal keys before writing
    existing.pop("_config_file", None)

    config_path.write_text(json.dumps(existing, indent=2))
    return config_path


def calibrate(
    memory: Any,
    cfg: dict[str, Any],
    n: int = 200,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """
    Run calibration. Returns the vocab_map produced.

    Writes to config file unless dry_run=True.
    Raises RuntimeError on LLM call failure (does not write partial results).
    """
    from fidelis.recall import _resolve_filter_endpoint  # avoid circular at module level

    endpoint, token = _resolve_filter_endpoint(cfg)
    if not endpoint:
        raise RuntimeError(
            "No filter endpoint configured. Set COGITO_FILTER_ENDPOINT + "
            "COGITO_FILTER_TOKEN, or ANTHROPIC_API_KEY."
        )

    # calibrate_model overrides filter_model — calibration benefits from a larger model
    model = cfg.get("calibrate_model", cfg.get("filter_model", "anthropic/claude-haiku-4-5"))
    # Calibration runs once and has a large prompt — always use at least 90s
    timeout = max(cfg.get("filter_timeout_ms", 30000), 90000) / 1000
    user_id = cfg.get("user_id", "agent")

    print(f"[cogito calibrate] Sampling memories (n={n})...")
    memories = _sample_memories(memory, user_id, n)
    if not memories:
        raise RuntimeError("No memories in store. Run `cogito seed` first.")
    print(f"[cogito calibrate] Sampled {len(memories)} memories. Calling {model}...")

    vocab_map = _build_vocab_map(memories, endpoint, token, model, timeout)
    print(f"[cogito calibrate] {len(vocab_map)} vocab mappings extracted.")

    if dry_run:
        print("[cogito calibrate] DRY RUN — not writing config.")
        for k, v in list(vocab_map.items())[:20]:
            print(f"  {k!r:30s} → {v}")
        if len(vocab_map) > 20:
            print(f"  ... ({len(vocab_map) - 20} more)")
        return vocab_map

    path = _write_vocab_map(vocab_map, cfg)
    print(f"[cogito calibrate] Written to {path}")
    return vocab_map
