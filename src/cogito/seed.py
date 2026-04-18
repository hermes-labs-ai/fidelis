"""
cogito seed — bulk-seed the memory store from markdown/text files.

The agent (or a capable LLM) decides what to remember. Raw text is read,
a curation LLM (default: same filter endpoint as /recall) extracts a list
of atomic facts, and each fact is written verbatim via POST /store — no
mem0 extraction prompt involved.

Usage:
    cogito seed ~/memory/                          # seed all .md files
    cogito seed ~/memory/ ~/notes/sessions/        # multiple dirs
    cogito seed ~/memory/ --dry-run                # show facts without writing
    cogito seed ~/memory/ --force                  # re-seed even unchanged files
    cogito seed ~/memory/ --glob "*.md"            # filter by pattern
    cogito seed ~/memory/ --add                    # use /add (mem0 extraction) instead

State is tracked in ~/.cogito/seeded.json (file path → mtime hash).
Re-run at any time — only changed or new files are seeded.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# ── chunking ───────────────────────────────────────────────────────────────

def _chunks_from_file(path: Path, max_chars: int = 3000) -> list[str]:
    """
    Split a markdown file into chunks by heading.
    Larger chunks than before — the curation LLM reads the full section
    and decides which facts matter, so we want enough context per chunk.
    """
    import re
    text = path.read_text(errors="replace")
    if not text.strip():
        return []

    sections = re.split(r"(?m)^(?=#{1,3} )", text)
    sections = [s.strip() for s in sections if s.strip()]

    chunks: list[str] = []
    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            paras = re.split(r"\n{2,}", section)
            buf = ""
            for para in paras:
                para = para.strip()
                if not para:
                    continue
                if buf and len(buf) + len(para) + 2 > max_chars:
                    chunks.append(buf)
                    buf = para
                else:
                    buf = (buf + "\n\n" + para).strip() if buf else para
            if buf:
                chunks.append(buf)

    return [c for c in chunks if len(c.strip()) >= 40]


# ── LLM curation ───────────────────────────────────────────────────────────

_CURATE_SYSTEM = (
    "You are a memory curator for an AI agent. "
    "Read the provided text and extract a list of atomic, self-contained facts "
    "worth remembering for future reference. Each fact must be independently "
    "understandable without the surrounding text. "
    "Focus on: decisions made, tools/versions/configs, bugs fixed, "
    "architecture choices, project names, people, deadlines, lessons learned, "
    "file paths, API shapes, and anything that would be useful weeks later. "
    "Discard: status updates that are now stale, meeting chatter with no outcome, "
    "and anything too vague to be actionable. "
    "Output ONLY a JSON array of strings — one fact per string, no explanation. "
    'Example: ["lintlang v0.3.1 published to PyPI", "mem0 v1.0.5 reads from payload[\\"data\\"]"]'
)


def _curate(text: str, endpoint: str, token: str, model: str, timeout: float) -> list[str]:
    """
    Call the filter LLM to extract curated facts from a text chunk.
    Returns list of fact strings, or [] on failure.
    """
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _CURATE_SYSTEM},
            {"role": "user", "content": f"Text to extract facts from:\n\n{text}"},
        ],
        "max_tokens": 1000,
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

        # Strip thinking tokens
        if "<think>" in raw:
            end_think = raw.rfind("</think>")
            raw = raw[end_think + 8:].strip() if end_think >= 0 else raw

        # Some models (qwen) emit one ["fact"] per line instead of a single array.
        # Extract every [...] block and flatten.
        import re
        all_facts: list[str] = []
        for m in re.finditer(r"\[([^\[\]]*)\]", raw):
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, list):
                    all_facts.extend(f for f in parsed if isinstance(f, str) and len(f.strip()) > 5)
            except json.JSONDecodeError:
                pass

        return all_facts
    except Exception:
        return []


def _resolve_curation_endpoint(cfg: dict) -> tuple[str, str, str]:
    """Return (base_url, token, model) for the curation LLM."""
    endpoint = cfg.get("filter_endpoint", "")
    token = cfg.get("filter_token", "")
    model = cfg.get("curation_model", cfg.get("llm_model", "mistral:7b"))
    if endpoint and token:
        return endpoint.rstrip("/"), token, model

    api_key = cfg.get("anthropic_api_key", "")
    if api_key:
        return "https://api.anthropic.com", api_key, "claude-haiku-4-5-20251001"

    return "", "", model


# ── state tracking ──────────────────────────────────────────────────────────

def _state_path() -> Path:
    p = Path.home() / ".cogito" / "seeded.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_state() -> dict[str, str]:
    p = _state_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict[str, str]) -> None:
    _state_path().write_text(json.dumps(state, indent=2))


def _file_hash(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


# ── HTTP ────────────────────────────────────────────────────────────────────

def _store(base_url: str, text: str, timeout: int = 30) -> str:
    """POST /store — verbatim write. Returns memory id."""
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        f"{base_url}/store",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    return result.get("id", "")


def _add(base_url: str, text: str, timeout: int = 120) -> tuple[int, list[str]]:
    """POST /add — mem0 extraction path. Returns (count, memories)."""
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        f"{base_url}/add",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    return result.get("count", 0), result.get("memories", [])


def _check_server(base_url: str) -> int:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
            return json.loads(resp.read()).get("count", 0)
    except urllib.error.URLError as e:
        raise RuntimeError(f"cogito server not reachable at {base_url}") from e


# ── core ────────────────────────────────────────────────────────────────────

def seed(
    sources: list[Path],
    base_url: str,
    cfg: dict | None = None,
    glob_pattern: str = "*.md",
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = False,
    delay_ms: int = 0,
    use_add: bool = False,
) -> dict:
    """
    Seed the cogito store from source dirs/files.

    Default path: LLM curates facts → POST /store (verbatim, no extraction).
    With use_add=True: raw chunks → POST /add (mem0 extraction, legacy path).
    """
    cfg = cfg or {}
    state = _load_state()
    stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "chunks_read": 0,
        "facts_written": 0,
        "errors": 0,
    }

    # Curation endpoint
    curation_available = False
    if not use_add:
        endpoint, token, model = _resolve_curation_endpoint(cfg)
        timeout = cfg.get("filter_timeout_ms", 15000) / 1000
        if endpoint:
            curation_available = True
            print(f"[cogito seed] Curation model: {model} @ {endpoint}")
        else:
            print("[cogito seed] No curation endpoint — falling back to /add (mem0 extraction)")
            use_add = True

    # Collect files
    all_files: list[Path] = []
    for src in sources:
        src = src.expanduser().resolve()
        if src.is_file():
            all_files.append(src)
        elif src.is_dir():
            all_files.extend(sorted(src.rglob(glob_pattern)))
        else:
            print(f"  [skip] not found: {src}", file=sys.stderr)

    if not all_files:
        print("No files found to seed.")
        return stats

    if not dry_run:
        try:
            count_before = _check_server(base_url)
            print(f"[cogito seed] Server OK — {count_before} memories before seeding")
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    mode = "/add (mem0 extraction)" if use_add else "/store (agent-curated, verbatim)"
    print(f"[cogito seed] {len(all_files)} file(s) — write path: {mode}\n")

    for path in all_files:
        file_key = str(path)
        file_hash = _file_hash(path)

        if not force and state.get(file_key) == file_hash:
            stats["files_skipped"] += 1
            if verbose:
                print(f"  [=] {path.name}  (unchanged)")
            continue

        chunks = _chunks_from_file(path)
        if not chunks:
            state[file_key] = file_hash
            continue

        print(f"  [→] {path.name}  {len(chunks)} chunk(s)")
        stats["chunks_read"] += len(chunks)

        file_facts = 0
        file_errors = 0

        for i, chunk in enumerate(chunks, 1):
            if use_add:
                # Legacy: raw chunk → mem0 extraction → store
                if dry_run:
                    print(f"      [{i}/{len(chunks)}] DRY /add: {chunk[:80].replace(chr(10),' ')!r}")
                    stats["facts_written"] += 1
                    continue
                try:
                    count, memories = _add(base_url, chunk)
                    file_facts += count
                    stats["facts_written"] += count
                    if verbose:
                        for m in memories:
                            print(f"      + {m[:90]}")
                    if delay_ms:
                        time.sleep(delay_ms / 1000)
                except Exception as e:
                    print(f"      [!] chunk {i}: {e}", file=sys.stderr)
                    file_errors += 1
                    stats["errors"] += 1
            else:
                # Preferred: LLM curates → /store verbatim
                facts = _curate(chunk, endpoint, token, model, timeout)  # type: ignore[possibly-undefined]
                if not facts:
                    if verbose:
                        print(f"      [{i}/{len(chunks)}] (no facts extracted)")
                    continue

                if dry_run:
                    for f in facts:
                        print(f"      DRY: {f[:100]}")
                    stats["facts_written"] += len(facts)
                    continue

                for fact in facts:
                    try:
                        _store(base_url, fact)
                        file_facts += 1
                        stats["facts_written"] += 1
                        if verbose:
                            print(f"      + {fact[:100]}")
                        if delay_ms:
                            time.sleep(delay_ms / 1000)
                    except Exception as e:
                        print(f"      [!] store failed: {e}", file=sys.stderr)
                        file_errors += 1
                        stats["errors"] += 1

        if not dry_run and file_errors == 0:
            state[file_key] = file_hash

        stats["files_processed"] += 1
        print(f"      done — +{file_facts} facts  ({file_errors} errors)")

    if not dry_run:
        _save_state(state)
        try:
            count_after = _check_server(base_url)
            print(f"\n[cogito seed] Done. {count_after} memories in store.")
        except Exception:
            pass

    print(f"\n  files processed : {stats['files_processed']}")
    print(f"  files skipped   : {stats['files_skipped']}  (unchanged)")
    print(f"  chunks read     : {stats['chunks_read']}")
    print(f"  facts written   : {stats['facts_written']}")
    if stats["errors"]:
        print(f"  errors          : {stats['errors']}")

    return stats
