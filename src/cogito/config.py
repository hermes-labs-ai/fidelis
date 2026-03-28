"""
cogito config — loaded from env vars or .cogito.json.

Priority: env vars > .cogito.json > defaults.
No workspace paths. No internal tooling assumptions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_TECHNICAL_EXTRACTION_PROMPT = (
    "You are a Technical Memory Organizer for an AI agent workspace. "
    "Extract ALL factual statements from the input — technical decisions, "
    "infrastructure changes, product details, research findings, scores, dates, "
    "names, URLs, versions, and architectural choices.\n\n"
    "Types of information to extract:\n"
    "1. Infrastructure changes (services started/stopped, configs changed, versions installed)\n"
    "2. Technical decisions and their rationale\n"
    "3. Product/project details (names, versions, stars, URLs, status)\n"
    "4. Research findings and experiment results (scores, percentages, benchmarks)\n"
    "5. People, organizations, and relationships\n"
    "6. Dates, deadlines, and timelines\n"
    "7. File paths, port numbers, model names, and system specifics\n"
    "8. Security findings and audit scores\n"
    "9. Architecture patterns and design decisions\n"
    "10. Task assignments and status changes\n\n"
    "Examples:\n\n"
    "Input: We upgraded the server to Ubuntu 24.04 LTS\n"
    "Output: {\"facts\": [\"Server upgraded to Ubuntu 24.04 LTS\"]}\n\n"
    "Input: Recall@1 improved from 58% to 85% after adding the snapshot layer\n"
    "Output: {\"facts\": [\"Recall@1 improved from 58% to 85% after snapshot layer addition\"]}\n\n"
    "Input: Hi, how are you?\n"
    "Output: {\"facts\": []}\n\n"
    "Return facts as JSON with key \"facts\" and a list of strings. "
    "Extract generously — it's better to capture too much than too little. "
    "Every concrete fact matters."
)

_DEFAULTS: dict[str, Any] = {
    "port": 19420,
    "user_id": "agent",
    "recall_limit": 50,
    "recall_threshold": 400.0,
    "query_threshold": 250.0,
    "filter_model": "anthropic/claude-haiku-4-5",
    "filter_timeout_ms": 12000,
    "store_path": str(Path.home() / ".cogito" / "store"),
    "collection": "cogito_memory",
    "ollama_url": "http://localhost:11434",
    "llm_model": "mistral:7b",
    "embed_model": "nomic-embed-text",
    "vocab_map": {},
    "calibrate_model": "mistral:7b",
    "custom_fact_extraction_prompt": _TECHNICAL_EXTRACTION_PROMPT,
}

_ENV_MAP = {
    "COGITO_PORT": ("port", int),
    "COGITO_USER_ID": ("user_id", str),
    "COGITO_FILTER_ENDPOINT": ("filter_endpoint", str),
    "COGITO_FILTER_TOKEN": ("filter_token", str),
    "COGITO_FILTER_MODEL": ("filter_model", str),
    "COGITO_FILTER_TIMEOUT_MS": ("filter_timeout_ms", int),
    "COGITO_STORE_PATH": ("store_path", str),
    "COGITO_COLLECTION": ("collection", str),
    "COGITO_OLLAMA_URL": ("ollama_url", str),
    "COGITO_LLM_MODEL": ("llm_model", str),
    "COGITO_EMBED_MODEL": ("embed_model", str),
    "COGITO_RECALL_LIMIT": ("recall_limit", int),
    "COGITO_RECALL_THRESHOLD": ("recall_threshold", float),
    "COGITO_QUERY_THRESHOLD": ("query_threshold", float),
    # Also accept raw Anthropic key for direct calls (no gateway needed)
    "ANTHROPIC_API_KEY": ("anthropic_api_key", str),
}


def load(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load config. Searches for .cogito.json in cwd and home dir if not specified.
    Env vars override file values.
    """
    cfg: dict[str, Any] = dict(_DEFAULTS)

    # File
    paths_to_try: list[Path] = []
    if config_path:
        paths_to_try.append(Path(config_path))
    else:
        paths_to_try += [
            Path.cwd() / ".cogito.json",
            Path.home() / ".cogito" / "config.json",
        ]

    for p in paths_to_try:
        if p.exists():
            try:
                with open(p) as f:
                    file_cfg = json.load(f)
                cfg.update(file_cfg)
                cfg["_config_file"] = str(p)
                break
            except Exception:
                pass

    # Env vars win
    for env_key, (cfg_key, cast) in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                cfg[cfg_key] = cast(val)
            except (ValueError, TypeError):
                pass

    return cfg


def mem0_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the mem0 Memory.from_config() dict from resolved config."""
    store_path = str(Path(cfg["store_path"]).expanduser())

    m = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": cfg["llm_model"],
                "ollama_base_url": cfg["ollama_url"],
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": cfg["embed_model"],
                "ollama_base_url": cfg["ollama_url"],
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": cfg["collection"],
                "path": store_path,
            },
        },
    }

    if "custom_fact_extraction_prompt" in cfg:
        m["custom_fact_extraction_prompt"] = cfg["custom_fact_extraction_prompt"]

    return m
