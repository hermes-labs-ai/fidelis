"""Bind the Gemini CLI extension manifest to the checked-in release truth.

``gemini-extension.json`` is what ``gemini extensions install
https://github.com/hermes-labs-ai/fidelis`` and the geminicli.com gallery
crawler read. It must launch the same published package that ``server.json``
advertises to the MCP Registry, at the same version, so a release bump cannot
leave one surface pointing at an older wheel.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "gemini-extension.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _package_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE).group(1)


def test_manifest_is_at_repository_root_with_gallery_safe_name():
    manifest = _manifest()
    assert MANIFEST.parent == ROOT
    # Gemini CLI: lowercase letters, digits, and dashes; the name is the
    # extension directory name users see in `gemini extensions list`.
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", manifest["name"])
    assert manifest["name"] == "fidelis"
    assert manifest["description"].strip()
    assert manifest["contextFileName"] == "GEMINI.md"
    assert (ROOT / manifest["contextFileName"]).is_file()


def test_manifest_version_matches_package_and_registry_manifest():
    manifest = _manifest()
    server = json.loads((ROOT / "server.json").read_text())
    assert manifest["version"] == _package_version()
    assert manifest["version"] == server["version"]


def test_manifest_launches_the_registry_published_package():
    manifest = _manifest()
    server = json.loads((ROOT / "server.json").read_text())
    package = server["packages"][0]
    assert package["registryType"] == "pypi"

    entry = manifest["mcpServers"]["fidelis"]
    assert entry["command"] == package["runtimeHint"] == "uvx"
    assert "trust" not in entry  # unsupported inside extensions

    runtime = package["runtimeArguments"][0]
    package_args = [argument["value"] for argument in package["packageArguments"]]
    expected = [
        runtime["name"],
        f'{runtime["value"]}=={package["version"]}',
        *package_args,
    ]
    assert entry["args"] == expected
    # No `${extensionPath}` reference: nothing in the repository is executed,
    # so a gallery install needs only `uv` on PATH.
    assert not any("${extensionPath}" in arg for arg in entry["args"])


def test_context_file_names_only_tools_the_server_exposes():
    from fidelis.mcp_server import TOOLS

    exposed = {tool["name"] for tool in TOOLS}
    named = set(re.findall(r"`(fidelis_[a-z_]+)`", (ROOT / "GEMINI.md").read_text()))
    assert named, "GEMINI.md should tell the model which tools exist"
    assert named <= exposed, named - exposed


def test_readme_documents_the_extension_install_command():
    readme = (ROOT / "README.md").read_text()
    assert "gemini extensions install https://github.com/hermes-labs-ai/fidelis" in readme
    assert "gemini-extension.json" in readme
