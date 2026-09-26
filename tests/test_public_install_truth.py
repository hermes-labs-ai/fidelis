"""Keep public install and evidence links bound to this repository."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SURFACES = (
    ROOT / "llms.txt",
    ROOT / "agents.md",
    ROOT / "docs" / "full-reference.md",
)
RELEASE_NOTES = ROOT / "docs" / "releases" / "0.3.0rc1.md"


def _package_version() -> str:
    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        (ROOT / "pyproject.toml").read_text(),
        re.MULTILINE,
    )
    assert match is not None
    return match.group(1)


def test_public_surfaces_do_not_install_unrelated_pypi_project():
    for path in PUBLIC_SURFACES:
        text = path.read_text()
        assert "pip install fidelis\n" not in text, path
        assert "pip install fidelis " not in text, path
        assert not re.search(r"pypi\.org/project/fidelis(?!-memory)\b", text), path


def test_primary_surfaces_install_the_fidelis_memory_distribution():
    version = _package_version()
    for path in (ROOT / "llms.txt", ROOT / "docs" / "full-reference.md"):
        text = path.read_text()
        assert re.search(
            rf'python3 -m pip install "fidelis-memory(?:\[hybrid\])?=={re.escape(version)}"',
            text,
        ), path


def test_python_and_citation_versions_match_package():
    version = _package_version()
    package_init = (ROOT / "src" / "fidelis" / "__init__.py").read_text()
    citation = (ROOT / "CITATION.cff").read_text()
    assert f'__version__ = "{version}"' in package_init
    assert f'version: "{version}"' in citation


def test_current_release_notes_and_container_metadata_track_package_version():
    """Bind release-facing, non-historical surfaces to the package version."""
    version = _package_version()
    notes = RELEASE_NOTES.read_text()
    assert notes.startswith(f"<!-- release-version: {version} -->\n")
    assert f"# Fidelis Memory {version} " in notes
    assert f'fidelis-memory=={version}' in notes

    dockerfile = (ROOT / "Dockerfile").read_text()
    compose = (ROOT / "docker-compose.yml").read_text()
    assert f'org.opencontainers.image.version="{version}"' in dockerfile
    assert f"image: fidelis:{version}" in compose


def test_registry_manifest_pins_the_released_distribution():
    manifest = json.loads((ROOT / "server.json").read_text())
    package = manifest["packages"][0]
    assert package["version"] == manifest["version"] == _package_version()
    assert package["registryType"] == "pypi"
    assert package["runtimeHint"] == "uvx"
    assert package["runtimeArguments"] == [
        {"type": "named", "name": "--from", "value": f'fidelis-memory=={package["version"]}'}
    ]
    assert [arg["value"] for arg in package["packageArguments"]] == ["fidelis", "mcp", "serve"]


def test_release_surfaces_do_not_recycle_historical_scores():
    for name in ("llms.txt", "agents.md"):
        text = (ROOT / name).read_text()
        assert "83.2%" not in text
        assert "73.0%" not in text
