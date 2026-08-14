"""Keep CodeMeta synchronized with the tracked release metadata."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _toml_string(text, section, key):
    section_match = re.search(
        rf"(?ms)^\[{re.escape(section)}\]\n(?P<body>.*?)(?=^\[|\Z)", text
    )
    assert section_match, section
    value_match = re.search(
        rf'(?m)^{re.escape(key)}\s*=\s*"([^"]+)"\s*$', section_match.group("body")
    )
    assert value_match, f"{section}.{key}"
    return value_match.group(1)


def test_codemeta_matches_release_metadata():
    codemeta = json.loads((ROOT / "codemeta.json").read_text())
    pyproject = (ROOT / "pyproject.toml").read_text()

    distribution = _toml_string(pyproject, "project", "name")
    repository = _toml_string(pyproject, "project.urls", "Repository")
    issues = _toml_string(pyproject, "project.urls", "Issues")
    version = _toml_string(pyproject, "project", "version")

    assert codemeta["@context"] == "https://w3id.org/codemeta/3.1"
    assert codemeta["@type"] == "SoftwareSourceCode"
    assert codemeta["name"] == "fidelis"
    assert distribution == "fidelis-memory"
    assert codemeta["version"] == version
    assert codemeta["codeRepository"] == repository
    assert codemeta["url"] == repository
    assert codemeta["issueTracker"] == issues
    assert codemeta["downloadUrl"] == (
        f"https://pypi.org/project/{distribution}/{version}/"
    )
    assert codemeta["releaseNotes"] == f"{repository}/releases/tag/v{version}"
