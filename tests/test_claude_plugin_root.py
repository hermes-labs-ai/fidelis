"""Bind the portable plugin root to the distribution it actually installs.

The plugin root ships no product code: it is a manifest, an MCP connection
launched from PyPI, and a skill. Everything that can silently drift between it
and the package — the version pin, the distribution name, the tool surface it
advertises — is asserted here.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "claude-plugin"
MANIFEST = PLUGIN / ".claude-plugin" / "plugin.json"
MCP_CONFIG = PLUGIN / ".mcp.json"
SKILL = PLUGIN / "skills" / "fidelis-memory" / "SKILL.md"

DISTRIBUTION = "fidelis-memory"


def _package_version() -> str:
    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        (ROOT / "pyproject.toml").read_text(),
        re.MULTILINE,
    )
    assert match is not None
    return match.group(1)


def test_plugin_root_has_the_three_required_files():
    for path in (MANIFEST, MCP_CONFIG, SKILL):
        assert path.is_file(), path


def test_manifest_version_tracks_the_package_version():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["name"] == "fidelis"
    assert manifest["version"] == _package_version()


def test_mcp_command_pins_the_released_distribution_exactly():
    """A floating pin would install a different server than the one tested."""
    servers = json.loads(MCP_CONFIG.read_text())["mcpServers"]
    assert list(servers) == ["fidelis"]
    entry = servers["fidelis"]
    assert entry["command"] == "uvx"
    assert entry["args"] == [
        "--from",
        f"{DISTRIBUTION}=={_package_version()}",
        "fidelis",
        "mcp",
        "serve",
    ]


def test_mcp_command_matches_the_registry_manifest():
    """server.json is the MCP Registry contract; the plugin must not diverge.

    The two spell the version differently on purpose: the registry carries it
    in `packages[].version` and leaves `--from` bare, while a host reading
    `.mcp.json` gets a literal argv and needs the pin inlined. Same resolved
    command either way, so both halves are checked against one source.
    """
    package = json.loads((ROOT / "server.json").read_text())["packages"][0]
    entry = json.loads(MCP_CONFIG.read_text())["mcpServers"]["fidelis"]

    assert package["identifier"] == DISTRIBUTION
    assert package["version"] == _package_version()
    assert package["runtimeHint"] == entry["command"]
    assert package["transport"]["type"] == "stdio"

    runtime_args = [(a["name"], a["value"]) for a in package["runtimeArguments"]]
    assert runtime_args == [("--from", DISTRIBUTION)]
    assert entry["args"][:2] == ["--from", f"{DISTRIBUTION}=={package['version']}"]

    assert [a["value"] for a in package["packageArguments"]] == entry["args"][2:]


def test_plugin_root_carries_no_product_code():
    """The catalog pins a commit; shipping code here would fork the package."""
    offenders = [p for p in PLUGIN.rglob("*.py") if p.is_file()]
    assert offenders == []


def test_skill_declares_the_tools_the_mcp_connection_provides():
    text = SKILL.read_text()
    assert text.startswith("---\n")
    front_matter = text.split("---\n", 2)[1]
    assert re.search(r"^name:\s*fidelis-memory\s*$", front_matter, re.MULTILINE)
    assert re.search(r"^description:\s*\S", front_matter, re.MULTILINE)
    for tool in ("fidelis_orient", "fidelis_recall", "fidelis_query", "fidelis_health"):
        assert tool in text, tool


def test_skill_does_not_promise_a_write_tool_over_mcp():
    """The MCP surface is read-only; only the HTTP API writes."""
    from fidelis.mcp_server import TOOLS

    names = {tool["name"] for tool in TOOLS}
    assert names == {"fidelis_orient", "fidelis_recall", "fidelis_query", "fidelis_health"}
    assert not any("store" in name or "add" in name for name in names)
