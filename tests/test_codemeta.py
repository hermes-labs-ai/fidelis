"""Keep CodeMeta synchronized with the tracked release metadata."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOI = "https://doi.org/10.5281/zenodo.21873318"
ORCID = "https://orcid.org/0009-0005-4896-1112"
VOLATILE_DATE_KEYS = {"dateCreated", "dateModified", "datePublished"}


def _toml_section(text, section):
    section_match = re.search(
        rf"(?ms)^\[{re.escape(section)}\]\n(?P<body>.*?)(?=^\[|\Z)", text
    )
    assert section_match, section
    return section_match.group("body")


def _toml_string(text, section, key):
    value_match = re.search(
        rf'(?m)^{re.escape(key)}\s*=\s*"([^"]+)"\s*$', _toml_section(text, section)
    )
    assert value_match, f"{section}.{key}"
    return value_match.group(1)


def _toml_inline_string(text, section, key, nested_key):
    table_match = re.search(
        rf"(?m)^{re.escape(key)}\s*=\s*\{{(?P<table>[^}}\n]+)\}}\s*$",
        _toml_section(text, section),
    )
    assert table_match, f"{section}.{key}"
    value_match = re.search(
        rf'\b{re.escape(nested_key)}\s*=\s*"([^"]+)"', table_match.group("table")
    )
    assert value_match, f"{section}.{key}.{nested_key}"
    return value_match.group(1)


def test_codemeta_matches_release_metadata():
    codemeta = json.loads((ROOT / "codemeta.json").read_text())
    pyproject = (ROOT / "pyproject.toml").read_text()
    citation = (ROOT / "CITATION.cff").read_text()

    distribution = _toml_string(pyproject, "project", "name")
    license_name = _toml_inline_string(pyproject, "project", "license", "text")
    repository = _toml_string(pyproject, "project.urls", "Repository")
    issues = _toml_string(pyproject, "project.urls", "Issues")
    requires_python = _toml_string(pyproject, "project", "requires-python")
    version = _toml_string(pyproject, "project", "version")

    assert codemeta["@context"] == "https://w3id.org/codemeta/3.1"
    assert codemeta["@type"] == "SoftwareSourceCode"
    assert codemeta["name"] == "fidelis"
    assert distribution == "fidelis-memory"
    assert distribution in codemeta["description"]
    assert codemeta["version"] == version
    assert codemeta["codeRepository"] == repository
    assert codemeta["url"] == repository
    assert codemeta["issueTracker"] == issues
    assert codemeta["license"] == f"https://spdx.org/licenses/{license_name}"
    assert codemeta["runtimePlatform"] == f"Python {requires_python}"
    assert codemeta["downloadUrl"] == (
        f"https://pypi.org/project/{distribution}/{version}/"
    )
    assert codemeta["releaseNotes"] == f"{repository}/releases/tag/v{version}"
    assert codemeta["identifier"] == DOI
    assert codemeta["author"][0]["@id"] == ORCID
    assert f'orcid: "{ORCID}"' in citation
    assert "applicationCategory" not in codemeta
    assert VOLATILE_DATE_KEYS.isdisjoint(codemeta)
